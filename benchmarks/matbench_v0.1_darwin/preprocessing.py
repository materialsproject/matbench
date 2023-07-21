import pandas as pd
import random
import json 
def convert_gap(train_inputs,train_outputs=None,train=True):
    input_df = pd.DataFrame(train_inputs)
    if train:
        output_df = pd.DataFrame(train_outputs)
        df = input_df.join(output_df)
    else:
        df = input_df
    
    slot = {"composition":"composition", 
            "band gap":"gap expt"}
    pair1 = [("band gap", "composition")]
    data_list = []
    for df_slice in df.iterrows():
        for p in pair1:
            tmp_dict = {}
            ran = random.randint(0,3)
            if ran == 0:
                question = "What is "+ p[0]+" of given "+ p[1] + "? ->"
            elif ran == 1:
                question = "Write "+ p[0]+ " of given "+ p[1] + ". ->"
            elif ran == 2:
                question = "Given " + p[1] + ", write its "+ p[0] + ". ->"
            elif ran == 3:
                question = "Tell me " + p[0] + " of given " + p[1] + ". ->"
            elif ran == 4:
                question = "Given " + p[1] + ", what is its "+ p[0] + "? ->"
            tmp_dict["instruction"] = question
            tmp_dict["input"] = str(df_slice[1][slot[p[1]]])+"\n"
            if train:
                tmp_dict["output"] = " "+str(df_slice[1][slot[p[0]]])+"\n"
            data_list.append(tmp_dict)
   
    return data_list

def convert_metal(train_inputs,train_outputs=None,train=True):
    input_df = pd.DataFrame(train_inputs)
    if train:
        output_df = pd.DataFrame(train_outputs)
        df = input_df.join(output_df)
    else:
        df = input_df
    # key=csv column name, value=slot in Q
    slot = {"composition":"composition"}
    data_list = []
    for s in slot.keys():
        for i, element in enumerate(df[s]):
            tmp_dict = {}
            ran = random.randint(0,2)
            if ran == 0:
                question = "Is composition metal? ->"
            elif ran == 1:
                question = "Is given composition metal? ->"
            elif ran == 2:
                question = "Given composition, is it metal? ->"
            
            tmp_dict["instruction"] = question
            tmp_dict["input"] =  " "+ element +"\n"
            if train:
                answer = df['is_metal'][i]
                if answer == True:
                    tmp_dict["output"] = f" Yes, {element} is metal.\n"
                elif answer == False:
                    tmp_dict["output"] = f" No, {element} is not metal.\n"
            data_list.append(tmp_dict)
    if train:
        # https://github.com/psobko/Common-English-Nouns
        with open('2325_nouns.json', 'r', encoding='utf-8') as f:
            nouns = json.load(f)
    
        # add no-answer question according to length of data_list
        add = int(len(data_list)/30)
        
        add_nouns = random.sample(nouns, add)
        
        for an in add_nouns:
            tmp_dict = {}
            ran1 = random.randint(0,2)
            ran2 = random.sample(list(slot.values()), 1)[0]
            if ran1 == 0:
                question = "Is composition metal? ->"
            elif ran1 == 1:
                question = "Is given composition metal? ->"
            elif ran1 == 2:
                question = "Given composition, is it metal? ->"
            tmp_dict["instruction"] = question
            tmp_dict["input"] = " " + an + "\n"
            if train:
                tmp_dict["output"] = " "+an+" is not a "+ran2+" and it is not metal.\n"
            data_list.append(tmp_dict)
        random.shuffle(data_list)
    return data_list

def convert_steels(train_inputs,train_outputs=None,train=True):
    input_df = pd.DataFrame(train_inputs)
    if train:
        output_df = pd.DataFrame(train_outputs)
        df = input_df.join(output_df)
    else:
        df = input_df
    # key=csv column name, value=slot in Q
    slot = {"composition":"composition"}
    data_list = []
    for s in slot.keys():
        for i, element in enumerate(df[s]):
            tmp_dict = {}
            ran = random.randint(0,2)

            # What will be yield strength of composition at 800-1200 °C 
            if ran == 0:
                question = "What will be the yield strength of given composition at 800-1200 °C? ->"
            elif ran == 1:
                question = "Write a possible yield strength of given composition at 800-1200 °C. ->"
            elif ran == 2:
                question = "Given composition, write its potential yield strength at 800-1200 °C. ->"
            
            tmp_dict["instruction"] = question
            tmp_dict["input"] = " "+ element +"\n"
            if train:
                answer = df['yield strength'][i]
                tmp_dict["output"] = " "+str(answer)+"\n"
            data_list.append(tmp_dict)
    if train:
        # https://github.com/psobko/Common-English-Nouns
        with open('2325_nouns.json', 'r', encoding='utf-8') as f:
            nouns = json.load(f)
        # add no-answer question according to length of data_list
        add_c = int(len(data_list)/50)
        add_n = int(len(data_list)/30)
        
        add_comps = random.sample(df['composition'].tolist(), add_c)
    

        for an in add_comps:
            tmp_dict = {}
            ran1 = random.randint(0,2)
            ran2 = random.sample(list(slot.values()), 1)[0]
            if ran1 == 0:
                question = "What is yield strength of composition? ->"
            elif ran1 == 1:
                question = "Write a possible yield strength of given composition. ->"
            elif ran1 == 2:
                question = "Given composition, write its potential yield strength. ->"
            tmp_dict["instruction"] = question
            tmp_dict["input"] = " "+ an + ".\n"
            if train:
                tmp_dict["output"] = " Unable to answer due to lack of conditions.\n"
            data_list.append(tmp_dict)
        # add no-answer question according to length of data_list
        add_nouns = random.sample(nouns, add_n)
    
        for an in add_nouns:
            tmp_dict = {}
            ran1 = random.randint(0,2)
            ran2 = random.sample(list(slot.values()), 1)[0]
            if ran1 == 0:
                question = "What will be the yield strength of given composition at 800-1200 °C? ->"
            elif ran1 == 1:
                question = "Write a possible yield strength of given composition at 800-1200 °C. ->"
            elif ran1 == 2:
                question = "Given composition, write its potential yield strength at 800-1200 °C. ->"
            tmp_dict["instruction"] = question
            tmp_dict["input"] = an
            if train:
                tmp_dict["output"] = " "+an+" is not a "+ran2+" and it does not have yield strength.\n"
            data_list.append(tmp_dict)
        random.shuffle(data_list)
    return data_list

def convert_glass(train_inputs,train_outputs=None,train=True):
    input_df = pd.DataFrame(train_inputs)
    if train:
        output_df = pd.DataFrame(train_outputs)
        df = input_df.join(output_df)
    else:
        df = input_df
    slot = {"composition":"composition", 
        "glass formation ability":"gfa"}
    pair1 = [("composition", "glass formation ability")]
    data_list = []
    for df_slice in df.iterrows():
        for p in pair1:
            tmp_dict = {}
            ran = random.randint(0,1)
            ran1 = random.randint(0,1)
            if ran1 == 0:
                form = "glass formation ability"
            elif ran1 == 1:
                form = "glass-forming ability"
            if ran == 0:
                question = "Does given "+ p[0]+" have "+ form + "? ->"
            elif ran == 1:
                question = "Tell me if given "+ p[0]+ " has "+ form + ". ->"
            tmp_dict["instruction"] = question
            tmp_dict["input"] = " "+str(df_slice[1][slot[p[0]]])+"\n"
            if train:
                if str(df_slice[1][slot[p[1]]]) == "True":
                    answer = " Yes, " + str(df_slice[1][slot[p[0]]]) + " has "+form+".\n"
                else:
                    answer = " No, "+ str(df_slice[1][slot[p[0]]]) + " does not have "+form+".\n"
                tmp_dict["output"] = answer
            data_list.append(tmp_dict)
    if train:
        # https://github.com/psobko/Common-English-Nouns
        with open('2325_nouns.json', 'r', encoding='utf-8') as f:
            nouns = json.load(f)
    
        # add no-answer question according to length of data_list
        add = int(len(data_list)/30)
        
        add_nouns = random.sample(nouns, add)
        
        for an in add_nouns:
            tmp_dict = {}
            ran = random.randint(0,1)
            ran1 = random.randint(0,1)
            if ran1 == 0:
                form = "glass formation ability"
            elif ran1 == 1:
                form = "glass-forming ability"
            if ran == 0:
                question = "Does given "+ p[0]+" have "+ form + "? ->"
            elif ran == 1:
                question = "Tell me if given "+ p[0]+ " has "+ form + ". ->"
            tmp_dict["instruction"] = question
            tmp_dict["input"] = " "+an+"\n"
            if train:
                tmp_dict["output"] = " "+an+" is not a composition and it has no relation with "+form+".\n"
            data_list.append(tmp_dict)
        random.shuffle(data_list)
    return data_list