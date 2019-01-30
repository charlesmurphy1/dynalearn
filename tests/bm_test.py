from dynalearn.model.param import *
from dynalearn.model.unit import *
from dynalearn.model.bm import *

if __name__ == '__main__':

    units_info = {"v": Unit_info("v", 3, "visible"),
                  "h1": Unit_info("h1", 5, "hidden"),
                  "h2": Unit_info("h2", 4, "hidden"),}
    params_info = ["v-h1", "v-h2", "h1-h2"]
    batchsize = 1

    bm = General_Boltzmann_Machine(units_info, params_info)

    for k in bm.params:
        print(f"Weight {k} size: ", bm.params[k].size())
    print("Energy of config: ", bm.energy(bm.mc_units))
    print("Visible key: ", bm.v_key)
    print("Hidden keys: ", bm.h_keys)