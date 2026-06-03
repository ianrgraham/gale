//! Named DG fields and the typed field registry (`FieldSet`).
//!
//! A [`Field`] stores one named, multi-component DG quantity (velocity, pressure,
//! the log-conformation tensor, …) in **exactly** the layout the existing
//! operators use: one `Vec<f64>` of length `ndof = n_elements · n_nodes` per
//! component, with DOF index `e·n_nodes + k`. That makes the bridge to the
//! validated kernels a borrow, not a copy — `Field::components()` hands back the
//! same `&[Vec<f64>]` slice shape that `Hyperbolic::rhs`, `ViscoelasticFlow::step`,
//! etc. already consume.
//!
//! [`FieldSet`] is the per-`State` registry. Fields are addressed by string name
//! at the API surface (ergonomic, Python-friendly) but resolved through a cached
//! integer [`FieldId`] internally (cheap, hot-loop-safe) — the resolution chosen
//! in `docs/api-design.md` open-question #4.

use std::collections::HashMap;

/// A stable handle to a field within one [`FieldSet`], resolved once from a name
/// and then used for O(1) access in hot paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FieldId(pub usize);

/// One named, multi-component DG field.
///
/// `components[c]` is the `c`-th scalar component over all `ndof` nodes. A scalar
/// field has `n_comp == 1`; a 2D velocity has `n_comp == 2`; a symmetric 2×2
/// tensor stored as `(xx, xy, yy)` has `n_comp == 3` (the convention the
/// viscoelastic code already uses).
#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub n_comp: usize,
    pub ndof: usize,
    components: Vec<Vec<f64>>,
}

impl Field {
    /// Allocate a zeroed field of `n_comp` components over `ndof` nodes.
    pub fn zeros(name: impl Into<String>, n_comp: usize, ndof: usize) -> Self {
        Self { name: name.into(), n_comp, ndof, components: vec![vec![0.0; ndof]; n_comp] }
    }

    /// Wrap an existing component layout (e.g. the output of an operator) as a
    /// field. Panics if the components are ragged or empty, since a field must
    /// have a single well-defined `ndof`.
    pub fn from_components(name: impl Into<String>, components: Vec<Vec<f64>>) -> Self {
        assert!(!components.is_empty(), "field must have ≥1 component");
        let ndof = components[0].len();
        assert!(
            components.iter().all(|c| c.len() == ndof),
            "ragged field components: all components must have the same ndof"
        );
        Self { name: name.into(), n_comp: components.len(), ndof, components }
    }

    /// Borrow the components in `[n_comp][ndof]` layout — the shape the existing
    /// operators take. Zero-copy bridge to the validated kernels.
    #[inline]
    pub fn components(&self) -> &[Vec<f64>] {
        &self.components
    }

    /// Mutably borrow the components in `[n_comp][ndof]` layout.
    #[inline]
    pub fn components_mut(&mut self) -> &mut [Vec<f64>] {
        &mut self.components
    }

    /// Borrow a single component.
    #[inline]
    pub fn component(&self, c: usize) -> &[f64] {
        &self.components[c]
    }

    /// Mutably borrow a single component.
    #[inline]
    pub fn component_mut(&mut self, c: usize) -> &mut [f64] {
        &mut self.components[c]
    }

    /// Overwrite all components from a matching `[n_comp][ndof]` layout (e.g. the
    /// result of an integrator step). Shapes must match exactly.
    pub fn assign(&mut self, components: &[Vec<f64>]) {
        assert_eq!(components.len(), self.n_comp, "component count mismatch");
        for (dst, src) in self.components.iter_mut().zip(components) {
            assert_eq!(src.len(), self.ndof, "ndof mismatch");
            dst.copy_from_slice(src);
        }
    }
}

/// The per-`State` field registry: an ordered set of named fields with cached
/// name → [`FieldId`] resolution.
#[derive(Clone, Debug, Default)]
pub struct FieldSet {
    fields: Vec<Field>,
    index: HashMap<String, usize>,
}

impl FieldSet {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new zeroed field and return its handle. Panics if a field with
    /// the same name already exists.
    pub fn add(&mut self, name: impl Into<String>, n_comp: usize, ndof: usize) -> FieldId {
        let name = name.into();
        assert!(!self.index.contains_key(&name), "duplicate field name `{name}`");
        let id = FieldId(self.fields.len());
        self.index.insert(name.clone(), id.0);
        self.fields.push(Field::zeros(name, n_comp, ndof));
        id
    }

    /// Insert an already-built field and return its handle.
    pub fn insert(&mut self, field: Field) -> FieldId {
        assert!(!self.index.contains_key(&field.name), "duplicate field name `{}`", field.name);
        let id = FieldId(self.fields.len());
        self.index.insert(field.name.clone(), id.0);
        self.fields.push(field);
        id
    }

    /// Resolve a name to its handle once; use the handle in hot loops.
    pub fn id(&self, name: &str) -> Option<FieldId> {
        self.index.get(name).map(|&i| FieldId(i))
    }

    /// Whether a field with this name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Number of registered fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Borrow a field by name.
    pub fn get(&self, name: &str) -> Option<&Field> {
        self.index.get(name).map(|&i| &self.fields[i])
    }

    /// Mutably borrow a field by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Field> {
        match self.index.get(name) {
            Some(&i) => Some(&mut self.fields[i]),
            None => None,
        }
    }

    /// Borrow a field by handle (no name lookup).
    #[inline]
    pub fn by_id(&self, id: FieldId) -> &Field {
        &self.fields[id.0]
    }

    /// Mutably borrow a field by handle (no name lookup).
    #[inline]
    pub fn by_id_mut(&mut self, id: FieldId) -> &mut Field {
        &mut self.fields[id.0]
    }

    /// Iterate over all fields in registration order.
    pub fn iter(&self) -> impl Iterator<Item = &Field> {
        self.fields.iter()
    }
}
