# Remaining Major Issues in `causalProblem`

## Open issues

1. **Broad exception handling hides failures**
   - Multiple code paths use bare `except:` and continue with `pass`/`nan`, which can silently mask model/data/solver bugs.
   - References:
     - `autobounds/causalProblem.py:384`
     - `autobounds/causalProblem.py:491`
     - `autobounds/causalProblem.py:501`
     - `autobounds/causalProblem.py:530`
     - `autobounds/causalProblem.py:593`

2. **`respect_to` mutates caller globals unsafely**
   - It injects symbols into caller module globals and unconditionally deletes them on exit.
   - This can clobber pre-existing names and create hard-to-debug side effects.
   - References:
     - `autobounds/causalProblem.py:50`
     - `autobounds/causalProblem.py:53`
     - `autobounds/causalProblem.py:66`

3. **Mutable default arguments across API**
   - Several methods use defaults like `[]`/`{}`; this is risky and should be replaced with `None` + local initialization.
   - References:
     - `autobounds/causalProblem.py:231`
     - `autobounds/causalProblem.py:246`
     - `autobounds/causalProblem.py:272`
     - `autobounds/causalProblem.py:302`
     - `autobounds/causalProblem.py:361`
     - `autobounds/causalProblem.py:656`
     - `autobounds/causalProblem.py:685`
     - `autobounds/causalProblem.py:715`
     - `autobounds/causalProblem.py:729`

## Fixed in this pass

- Estimand initialization/check mismatch:
  - `self.estimand` now initializes as `None`.
  - `solve()` now raises a clear `ValueError` if estimand is unset.
  - References:
    - `autobounds/causalProblem.py:294`
    - `autobounds/causalProblem.py:484`
