# Method-deepening package index

Status: **partial**  
Ready: MDP-01, MDP-02, MDP-03  
Blocked on real paired refresh data: MDP-04

Authoritative components:

- `bundle/formulation/method_formulation_manifest.json`
- `bundle/complexity/routing_complexity_manifest.json`
- `bundle/equivariance/route_equivariance_manifest.json`
- `bundle/inventory/method_deepening_inventory_manifest.json`
- `bundle/synthesis_v2/method_deepening_package_manifest.json`
- `routing_complexity.json`
- `refresh_replay_contract.json`

`bundle/synthesis/` is the first generated candidate. `bundle/synthesis_v2/` is
authoritative because it additionally pins the replay contract and builder
script. Neither directory was overwritten.

The package is intentionally not eligible for a complete method-deepening
claim until the deterministic original-host refresh export passes the formal
v2 gates. The absence of that export does not invalidate the already accepted
MECH-09R loss-level causal result.

No experiment 43/44/45 result and no `HANDOFF.md` content was changed.
