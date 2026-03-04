# Remaining Issues in `causalProblem` (2026-03-04 pass)

## Open issues (ordered by severity)

1. **`respect_to` can clobber caller globals**
   - It injects symbols like `p`, `E`, `solve` into caller globals and then unconditionally deletes them on exit.
   - If the caller already had any of these names, values are lost after the context exits.
   - References:
     - `autobounds/causalProblem.py:51`
     - `autobounds/causalProblem.py:53`
     - `autobounds/causalProblem.py:66`

2. **Mutable default arguments remain in helper/public entry points**
   - Defaults like `number_values={}` and `cond=[]` are shared objects across calls.
   - They are not heavily mutated today, but this is brittle and can introduce cross-call state bugs later.
   - References:
     - `autobounds/causalProblem.py:247`
     - `autobounds/causalProblem.py:282`
     - `autobounds/causalProblem.py:348`

3. **`solve()` has dead variable assignment (`ci`)**
   - `ci = kwargs.get("ci", False)` is currently unused in the orchestrator `solve` path.
   - This is not a runtime bug, but it creates misleading intent and should be removed or used.
   - Reference:
     - `autobounds/causalProblem.py:406`

4. **Bounder helper coupling remains circular/fragile**
   - `Bounder.py` imports helper functions from `causalProblem.py`, while `causalProblem` imports `Bounder` dynamically.
   - It works now, but this file-level coupling makes refactors/import-order changes risky.
   - References:
     - `autobounds/Bounder.py:19`
     - `autobounds/causalProblem.py:283`
