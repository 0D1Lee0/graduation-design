import pickle
from const import OUTPUT_FOLDER
save_path = OUTPUT_FOLDER

itemcf_sim_path = save_path+'itemcf_i2i_sim_04-05.pkl'
itemcf_rr_sim_path = save_path+'itemcf_related_rule_i2i_sim.pkl'

itemcf_sim = pickle.load(open(itemcf_sim_path, 'rb'))
itemcf_rr_sim = pickle.load((open(itemcf_rr_sim_path, 'rb')))

flag = False
for i, item_sim in itemcf_sim.items():
    for j, sim in item_sim.items():
        if itemcf_rr_sim[i][j] != sim:
            flag = True
            print("i="+str(i)+", j="+str(j))
            print("sim="+str(sim)+", rr_sim="+str(itemcf_rr_sim[i][j]))
            break
    if flag:
        print("Not match!")
        break;