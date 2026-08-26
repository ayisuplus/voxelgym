# Voxel Core

Voxel Core defines the deterministic voxel world whose state can be recorded, restored, branched, and advanced under configurable physics.

## Language

**World Snapshot**:
The canonical, versioned byte representation of all restorable world state that can affect future simulation.
_Avoid_: Checkpoint, save file

**Checkpoint**:
A World Snapshot captured at a particular tick and embedded in a dataset for branching or replay verification.
_Avoid_: Snapshot format, save file
