# RDX upstream scanner audit

This is the implementation-oriented audit used for the RDX 1.0.0 final pass.
Repositories are treated as protocol, algorithm, schema, or workflow references;
historical payload forks are not assumed to be interchangeable.

## Highest-value current references

- [MemDBG](https://github.com/seregonwar/MemDBG): native PS4/PS5 daemon framing,
  capability discovery, memory maps/read/write, one-hop pointer seeds, and the
  combined scanner/debugger/trainer workflow. RDX uses capability-gated native
  TCP 9020 I/O and retains a compatibility fallback for early builds.
- [ps5debug-NG](https://github.com/OpenSourcereR-dev/ps5debug-NG): authoritative
  value-type IDs, exact/refine/AOB/TurboScan packet layouts, process maps, and
  debugger primitives. RDX now sends signed and float exact types using this
  ABI and falls back to typed host refinement where required.
- [ps5dbg](https://github.com/Darkatek7/ps5dbg): independent Python client
  confirmation of PS5Debug-NG signed/float packing, AOB masks, resident turbo
  sessions, and capability/fallback behavior.
- [PS5-MemoryPeeker](https://github.com/POWER-CHANGES-U/PS5-MemoryPeeker):
  concrete first/next `ScanEngine`, selected memory sections, typed value codec,
  bounded results, address batching, editable cheat rows, and export flow. RDX
  adopted an explicit Recommended/Writable/Readable scope and retained compact
  typed previous-value snapshots for relational scans.
- [PS4CheaterNeo](https://github.com/avan06/PS4CheaterNeo) and
  [ctn123/PS4_Cheater](https://github.com/ctn123/PS4_Cheater): established
  first/next/unknown/AOB workflow, section filtering, negative offsets,
  executable-root pointer scanning, result undo, and PointerFinder rescan
  behavior. RDX's target-relocation validation follows this model.
- [PINCE](https://github.com/korcankaraokcu/PINCE): pointer-map generation,
  explicit pointer-map comparison/filtering, signed offset bounds, terminal
  offset, depth/result limits, scope selection, and save/resume UX. RDX keeps
  its console-specific reverse index but now exposes its persisted cross-reload
  state as a Pointer Project.
- [Cheat Engine](https://github.com/cheat-engine/cheat-engine): mature reference
  for typed scans, pointer-map/rescan concepts, region controls, and separation
  between table editing and explicit memory writes. No Cheat Engine code was
  embedded in RDX.
- [GoldHEN Cheat Repository](https://github.com/GoldHEN/GoldHEN_Cheat_Repository),
  [GoldHEN Cheat Manager](https://github.com/GoldHEN/GoldHEN_Cheat_Manager), and
  [etaHEN PS5 Cheats](https://github.com/etaHEN/PS5_Cheats): end-user checkbox
  JSON uses raw `on`/`off` bytes at module-relative offsets. RDX emits this only
  for representable static patches and keeps pointer chains/live freezes native.

## Changes made from the comparison

- Replaced width-only scan semantics with `u8/i8/u16/i16/u32/i32/f32/u64/i64/
  f64/bytes` throughout scan, display, write, cheat, import, and export paths.
- Added wildcard AOB parsing (`AA BB ?? CC`), overlap-safe chunk scanning, and
  candidate-only next-scan validation.
- Added native PS5Debug-NG signed/float exact type IDs while preserving automatic
  host fallback and float tolerance refinement.
- Added Recommended game-region filtering to avoid obvious debugger, payload,
  and system-library noise without hiding Writable/Readable expert scopes.
- Retained previous typed values beside candidate addresses for changed,
  unchanged, increased, decreased, and known-delta scans.
- Exposed the mandatory two-relocation pointer validation state in the primary
  UI. Persisted records are scoped by process plus an ASLR-independent game
  image fingerprint.
- Replaced the single freeze worker with independent saved-cheat toggles.
- Added remembered endpoint/process/export preferences, reconnect, output
  selection, export preflight, atomic writes, and platform-correct deploy paths.

## Deliberate boundaries

- RDX does not label a same-session pointer chain permanent. It must survive two
  real target relocation epochs. Relaxing that rule would make trainer creation
  appear successful while exporting unstable heap paths.
- GoldHEN/etaHEN JSON has no RDX pointer-chain runtime and no PC-side continuous
  freeze loop. Dynamic entries stay in `*.rdx.json`; only module-relative raw
  patches are emitted to console-manager JSON.
- Watchpoints remain an explicit experimental action. They are not armed during
  ordinary scanning.
- The regression suite verifies algorithms, safety gates, serialization, and
  protocol framing. Physical-console behavior still depends on payload build,
  firmware, title layout, and network conditions and needs on-device testing.

## Historical and adjacent references

GoldHEN/ps4debug, jogolden/ps4debug, frame4, Reaper, PyPS4debug, Artemis,
MemoryEngine360, OpenOrbis, and older firmware-specific ps4debug forks were used
as lineage or architectural references. Dead, renamed, private, or unverified
repository names were not treated as authoritative current implementations.
