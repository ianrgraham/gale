//! HDF5 trajectory writer for gale DG simulations.
//!
//! One self-describing `.h5` file per run (no folders-of-text). Geometry and fields are
//! decoupled: the mesh is written **once** as a *topology* and only re-emitted when it
//! changes (AMR remesh); each frame stores the time-varying fields plus a reference to the
//! topology it uses. Moving rigid bodies store a pose per frame. Field datasets are stored as
//! `f32` (ample for visualization; compute stays `f64`), chunked and gzip-compressed.
//!
//! Layout:
//! ```text
//! /ref            attrs: order, dim
//! /topology/<id>/elem_nodes   [ne, nn, dim] f32   (per-element physical node coords)
//!               attrs: ne, nn
//! /frames/<NNNNNN>/           attrs: time, step, topology
//!     <field>     [ne, nn, ncomp] f32            (e.g. u, p, C)
//!     body_pose   [nbody, 3]      f64            (cx, cy, phi) — if any
//! ```
//! A field is *discontinuous across elements* (DG), so it is stored per-element (`[ne, nn, …]`)
//! and the viewer tessellates each element independently — faithfully showing inter-element jumps.

use gale::dg::Mesh2d;
use hdf5::File;

/// One time-varying field to record in a frame: `(name, flat [ne*nn*ncomp] f32 data, ncomp)`.
pub type FrameField<'a> = (&'a str, Vec<f32>, usize);

/// A registered topology's geometry, kept in memory so it can be re-emitted into a new file when a
/// size-capped trajectory rolls over (each file stays independently openable).
struct TopoData {
    ne: usize,
    nn: usize,
    nodes: Vec<f32>, // [ne*nn*2]
}

/// Append-style writer for a trajectory. Call [`write_mesh2d`](Self::write_mesh2d) once (or per
/// remesh) to register a topology, then [`write_frame`](Self::write_frame) per dumped step.
///
/// Single-file by default ([`create`](Self::create)); with [`create_split`](Self::create_split) the
/// writer caps each file at a byte budget and rolls to `<stem>.0001.h5`, `<stem>.0002.h5`, …,
/// re-emitting the live topology + body shapes as each new file's static state so every file opens
/// standalone (the viewer reads the whole `<stem>.NNNN.h5` set as one trajectory).
pub struct TrajectoryWriter {
    file: File,
    cur_path: String,
    stem: String,
    order: usize,
    dim: usize,
    max_bytes: Option<u64>,
    file_index: u32,
    frame: u64,         // global frame count across all files
    frame_in_file: u64, // resets each file (= the in-file frame-group index)
    topo_count: u64,
    topologies: Vec<TopoData>,
    body_radii: Option<Vec<f64>>,
}

impl TrajectoryWriter {
    /// Create a single-file trajectory at `path` and write the reference-element metadata.
    pub fn create(path: &str, order: usize, dim: usize) -> hdf5::Result<Self> {
        let file = File::create(path)?;
        Self::init_file(&file, order, dim)?;
        Ok(Self {
            file,
            cur_path: path.to_string(),
            stem: path.to_string(),
            order,
            dim,
            max_bytes: None,
            file_index: 0,
            frame: 0,
            frame_in_file: 0,
            topo_count: 0,
            topologies: Vec::new(),
            body_radii: None,
        })
    }

    /// Create a **size-capped, multi-file** trajectory: files are `<stem>.0000.h5`, `<stem>.0001.h5`,
    /// … each kept under `max_mb` megabytes. When a file passes the cap the writer rolls to the next,
    /// re-emitting the live topology + body shapes so each file is self-contained.
    pub fn create_split(stem: &str, order: usize, dim: usize, max_mb: u64) -> hdf5::Result<Self> {
        let cur_path = format!("{stem}.{:04}.h5", 0);
        let file = File::create(&cur_path)?;
        Self::init_file(&file, order, dim)?;
        Ok(Self {
            file,
            cur_path,
            stem: stem.to_string(),
            order,
            dim,
            max_bytes: Some(max_mb * (1 << 20)),
            file_index: 0,
            frame: 0,
            frame_in_file: 0,
            topo_count: 0,
            topologies: Vec::new(),
            body_radii: None,
        })
    }

    /// Write `/ref` metadata + the `topology`/`frames` groups into a freshly created file.
    fn init_file(file: &File, order: usize, dim: usize) -> hdf5::Result<()> {
        let r = file.create_group("ref")?;
        r.new_attr::<u64>().create("order")?.write_scalar(&(order as u64))?;
        r.new_attr::<u64>().create("dim")?.write_scalar(&(dim as u64))?;
        file.create_group("topology")?;
        file.create_group("frames")?;
        Ok(())
    }

    /// Write topology `id`'s geometry into `file` (used both on first registration and on roll-over).
    fn write_topo(file: &File, id: u64, t: &TopoData) -> hdf5::Result<()> {
        let g = file.group("topology")?.create_group(&id.to_string())?;
        g.new_dataset::<f32>()
            .deflate(4)
            .chunk((t.ne, t.nn, 2))
            .shape((t.ne, t.nn, 2))
            .create("elem_nodes")?
            .write_raw(&t.nodes)?;
        g.new_attr::<u64>().create("ne")?.write_scalar(&(t.ne as u64))?;
        g.new_attr::<u64>().create("nn")?.write_scalar(&(t.nn as u64))?;
        Ok(())
    }

    /// Register a 2D mesh as a new topology (per-element physical node coords), returning its id.
    /// Call once for a fixed mesh, or again after an AMR remesh; frames reference the id they use.
    pub fn write_mesh2d(&mut self, mesh: &Mesh2d) -> hdf5::Result<u64> {
        let nn = mesh.refq.n_nodes();
        let ne = mesh.n_elements();
        let mut nodes = vec![0f32; ne * nn * 2];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                nodes[(e * nn + k) * 2] = el.geom.x[k] as f32;
                nodes[(e * nn + k) * 2 + 1] = el.geom.y[k] as f32;
            }
        }
        let id = self.topo_count;
        let t = TopoData { ne, nn, nodes };
        Self::write_topo(&self.file, id, &t)?;
        self.topologies.push(t);
        self.topo_count += 1;
        Ok(id)
    }

    /// Register the rigid-body shapes once (disk radii, `[nbody]`), so the viewer can draw the
    /// bodies at each frame's `body_pose`. Static like the mesh — call once (radii don't change).
    pub fn write_body_radii(&mut self, radii: &[f64]) -> hdf5::Result<()> {
        Self::write_radii_to(&self.file, radii)?;
        self.body_radii = Some(radii.to_vec());
        Ok(())
    }

    fn write_radii_to(file: &File, radii: &[f64]) -> hdf5::Result<()> {
        let g = match file.group("bodies") {
            Ok(g) => g,
            Err(_) => file.create_group("bodies")?,
        };
        g.new_dataset::<f64>().shape(radii.len()).create("radius")?.write_raw(radii)?;
        Ok(())
    }

    /// Roll to the next file in a split trajectory: create `<stem>.NNNN.h5`, re-emit the live
    /// topology + body shapes as its static state, and reset the in-file frame counter.
    fn roll(&mut self) -> hdf5::Result<()> {
        self.file_index += 1;
        self.cur_path = format!("{}.{:04}.h5", self.stem, self.file_index);
        self.file = File::create(&self.cur_path)?;
        Self::init_file(&self.file, self.order, self.dim)?;
        for (id, t) in self.topologies.iter().enumerate() {
            Self::write_topo(&self.file, id as u64, t)?;
        }
        if let Some(r) = &self.body_radii {
            Self::write_radii_to(&self.file, r)?;
        }
        self.frame_in_file = 0;
        Ok(())
    }

    /// Write one frame: its fields (each `[ne, nn, ncomp]`) under a `topology` reference, plus
    /// optional rigid-body poses `(cx, cy, phi)`. `ne`/`nn` must match the referenced topology.
    /// In split mode, rolls to a new file first if the current one has passed the size cap.
    pub fn write_frame(
        &mut self,
        time: f64,
        step: u64,
        topo: u64,
        ne: usize,
        nn: usize,
        fields: &[FrameField],
        body_pose: Option<&[[f64; 3]]>,
    ) -> hdf5::Result<()> {
        if let Some(cap) = self.max_bytes {
            if self.frame_in_file > 0 {
                let sz = std::fs::metadata(&self.cur_path).map(|m| m.len()).unwrap_or(0);
                if sz >= cap {
                    self.roll()?;
                }
            }
        }
        let fg = self
            .file
            .group("frames")?
            .create_group(&format!("{:06}", self.frame_in_file))?;
        fg.new_attr::<f64>().create("time")?.write_scalar(&time)?;
        fg.new_attr::<u64>().create("step")?.write_scalar(&step)?;
        fg.new_attr::<u64>().create("topology")?.write_scalar(&topo)?;
        for (name, data, nc) in fields {
            assert_eq!(data.len(), ne * nn * nc, "field {name}: len != ne*nn*ncomp");
            fg.new_dataset::<f32>()
                .deflate(4)
                .chunk((ne, nn, *nc))
                .shape((ne, nn, *nc))
                .create(*name)?
                .write_raw(data)?;
        }
        if let Some(poses) = body_pose {
            let flat: Vec<f64> = poses.iter().flatten().copied().collect();
            fg.new_dataset::<f64>()
                .shape((poses.len(), 3))
                .create("body_pose")?
                .write_raw(&flat)?;
        }
        self.frame_in_file += 1;
        self.frame += 1;
        Ok(())
    }

    /// Total number of frames written across all files.
    pub fn n_frames(&self) -> u64 {
        self.frame
    }

    /// Number of files written (1 unless the size cap forced roll-overs).
    pub fn n_files(&self) -> u32 {
        self.file_index + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_writes_and_reads() {
        let path = std::env::temp_dir().join("gale_traj_test.h5");
        let p = path.to_str().unwrap();
        let (ne, nn) = (4usize, 9usize);
        {
            let mut w = TrajectoryWriter::create(p, 2, 2).unwrap();
            // fake topology written directly (no Mesh2d in the unit test)
            let g = w.file.group("topology").unwrap().create_group("0").unwrap();
            g.new_dataset::<f32>()
                .shape((ne, nn, 2))
                .create("elem_nodes")
                .unwrap()
                .write_raw(&vec![0f32; ne * nn * 2])
                .unwrap();
            w.topo_count = 1;
            let u: Vec<f32> = (0..ne * nn * 2).map(|i| i as f32).collect();
            w.write_frame(0.5, 1, 0, ne, nn, &[("u", u.clone(), 2)], None).unwrap();
            assert_eq!(w.n_frames(), 1);
        }
        let f = File::open(p).unwrap();
        let u: Vec<f32> = f.dataset("frames/000000/u").unwrap().read_raw().unwrap();
        assert_eq!(u.len(), ne * nn * 2);
        assert_eq!(u[5], 5.0);
        let t: f64 = f.group("frames/000000").unwrap().attr("time").unwrap().read_scalar().unwrap();
        assert_eq!(t, 0.5);
        std::fs::remove_file(p).ok();
    }
}
