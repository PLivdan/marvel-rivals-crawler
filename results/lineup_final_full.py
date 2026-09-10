"""After promotion: refit the LOCKED specification and penalties on all legal matches (development +
confirmation) for the published tables. The confirmation verdict refers to the development-row fit;
this refit only adds the 19,804 confirmed matches to the estimation sample."""
import json, sys, time, numpy as np
sys.path.insert(0, ".")
from lineup.fit import LineupDesign
lock = json.load(open("results/lineup_lock.json")); lams = lock["lams"]
d = LineupDesign("results/lineup_design"); w = np.ones(d.n)
fit = d.fit(lams, w, max_iter=10, verbose=True)
np.savez("results/lineup_published.npz", phi=fit["phi"], lams=np.array(lams), scalers=json.dumps(fit["scalers"]))
json.dump({"lams": lams, "n": int(d.n), "loglik": fit["loglik"], "n_iter": fit["n_iter"], "converged": fit["converged"], "scalers": fit["scalers"]}, open("results/lineup_published.json", "w"), indent=1)
print("published fit saved", fit["n_iter"], fit["converged"], flush=True)
