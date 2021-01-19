# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:15:03 2020

@author: sanky
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:10:25 2020

@author: sanky
"""

import random
from random import choice
import numpy as np
from search import *

class SET_THINGS:
    def __init__(self, thing):        
        self.objhold = thing
  

    def ObjP(self):
          return self.objhold
model={}        
cactus = "cactus"
flower = "flower"
goldenflower = "goldenflower"
D = "DOWN"
U = "UP"
L = "LEFT"
R = "RIGHT"

honeybee = "honeybee"

#adding all things into environment
set_things1_1 = SET_THINGS(honeybee)
set_things1_2 = SET_THINGS(None)
set_things1_3 = SET_THINGS(None)
set_things1_4 = SET_THINGS(cactus)
set_things1_5 = SET_THINGS(flower)
             
set_things2_1 = SET_THINGS(None)
set_things2_2 = SET_THINGS(cactus)
set_things2_3 = SET_THINGS(flower)
set_things2_4 = SET_THINGS(None)
set_things2_5 = SET_THINGS(cactus)
               
set_things3_1 = SET_THINGS(flower)
set_things3_2 = SET_THINGS(cactus)
set_things3_3 = SET_THINGS(None)
set_things3_4 = SET_THINGS(None)
set_things3_5 = SET_THINGS(None)
                
set_things4_1 = SET_THINGS(None)
set_things4_2 = SET_THINGS(flower)
set_things4_3 = SET_THINGS(cactus)
set_things4_4 = SET_THINGS(flower)
set_things4_5 = SET_THINGS(None)
                
set_things5_1 = SET_THINGS(None)
set_things5_2 = SET_THINGS(None)
set_things5_3 = SET_THINGS(None)
set_things5_4 = SET_THINGS(None)
set_things5_5 = SET_THINGS(goldenflower)

ENV =          [[set_things1_1,set_things1_2, set_things1_3, set_things1_4, set_things1_5],
               [set_things2_1, set_things2_2, set_things2_3, set_things2_4, set_things2_5],
               [set_things3_1, set_things3_2, set_things3_3, set_things3_4, set_things3_5],
               [set_things4_1, set_things4_2, set_things4_3, set_things4_4, set_things4_5],
               [set_things5_1, set_things5_2, set_things5_3, set_things5_4, set_things5_5]]

# Environment = [[HoneyBee,    None, 	       None ,       cactus,          flower         ],
#                [None ,       cactus,         flower,      None,            cactus         ],
#                [flower  ,    cactus,         None,        None,            None           ],
#                [None ,       flower,         cactus,      flower,          None           ],
#                [None ,       None,           None,        None,            goldenflower   ]]
               


#Reflex Agent
class HoneybeeReflexAgent:
    
    def __init__(self, env , honeybee_location , goal_location):
        self.env = env
        self.honeybee_location = honeybee_location
        self.goal_location = goal_location
        self.performance = 0
        self.checkgoalachived = False
        self.alive = True
        self.isGoalAchieved = False
        
    def isflower(self,honeybee_location):
        return self.env[honeybee_location[0]][honeybee_location[1]].ObjP == flower
    
    def iscactus(self,honeybee_location):
        return self.env[honeybee_location[0]][honeybee_location[1]].ObjP == cactus
        
    
    def nextDirection(self, honeybee_location):
        nextposition = []
        if honeybee_location[0] == 0:
            nextposition.append(D)
            if honeybee_location[1] == 0:
                nextposition.append(R)
            elif honeybee_location[1] == len(self.env[0])-1 :
                nextposition.append(L)
            else:
                nextposition.extend([L,R])

        elif honeybee_location[0] == len(self.env)-1 :
            nextposition.append(U)
            if honeybee_location[1] == len(self.env[0])-1:
                nextposition.append(L)
            elif honeybee_location[1] == 0:
                nextposition.append(R)
            else:
                nextposition.extend([L,R])

        elif honeybee_location[0] in range(1,len(self.env)) and honeybee_location[1] == 0:
            nextposition.extend([R, U, D])

        elif honeybee_location[0] in range(1,len(self.env)) and honeybee_location[1] == len(self.env[0]):
            nextposition.extend([R, U, D])

        else:
            nextposition.extend([U, D, L, R])

        nextDirection = random.choice(nextposition)

        next_location = tuple()

        if U == nextDirection:
            next_location= (honeybee_location[0]-1, honeybee_location[1])
        elif D == nextDirection:
            next_location = (honeybee_location[0]+1, honeybee_location[1])
        elif R == nextDirection:
            next_location = (honeybee_location[0], honeybee_location[1]+1)
        else:
            next_location = (honeybee_location[0],honeybee_location[1]-1)

        return next_location
    
    def perceptsin(self, honeybee_location):
        nextstate = self.env[honeybee_location[0]][honeybee_location[1]].ObjP()
        return nextstate

    def matchrules(self, currentstate):
        action = ""
        if currentstate in [cactus]:
            self.alive = False
            action = currentstate
        elif currentstate in [flower]: 
            self.alive = True
            action = "suck"
        else:
            action = currentstate
    
    def Reflexprogram(self, countofsteps):

        noofsteps = 0
        performance = 0

        while noofsteps < countofsteps and self.alive and not self.isGoalAchieved:
            next_location = self.nextDirection(self.honeybee_location)
            self.honeybee_location = next_location

            if next_location[0] == self.goal_location[0] and next_location[1] == self.goal_location[1]:
                self.isGoalAchieved = True
                print("HoneyBee suck the nectar from goldenflower at (",next_location[0],", ",next_location[1] ,")")
                
                
            print("HoneyBee flied to ", next_location)

            currentstate = self.perceptsin(next_location)
            action = self.matchrules(currentstate)

            if self.alive == False:
                print("HoneyBee Died : due to ", currentstate) 
                performance -= 20
            elif currentstate in [flower]:
                print("HoneyBee suck the nectar from flower : ","suck")
                performance += 50
            else:
                print("Nothing present in the location")
                performance -= 10
            noofsteps += 1


        print("Total steps performed : ", noofsteps)
        print("Performance : ", performance)
    
 

print("**************************** Reflex Agent **********************************************")

reflex = HoneybeeReflexAgent(ENV, [0,0], [5,5])
reflex.Reflexprogram(50)


print("****************************************************************************************")

#Model and Goal Based Agent
class HoneyBeeModelBasedAgent:
    def __init__(self, env, honeybee_location , goal_location):
        self.env = env
        self.honeybee_location = honeybee_location
        self.goal_location = goal_location      
        self.alive = True                
        self.isGoalAchieved = False
    
    
    
    
    def isflower(self,honeybee_location):
        return self.env[honeybee_location[0]][honeybee_location[1]].ObjP == flower
    
    def iscactus(self,honeybee_location):
        return self.env[honeybee_location[0]][honeybee_location[1]].ObjP == cactus
    
    def nextDirection(self, honeybee_location):
        nextposition = []
        if honeybee_location[0] == 0:
            nextposition.append(D)
            if honeybee_location[1] == 0:
                nextposition.append(R)
            elif honeybee_location[1] == len(self.env[0])-1 :
                nextposition.append(L)
            else:
                nextposition.extend([L,R])

        elif honeybee_location[0] == len(self.env)-1 :
            nextposition.append(U)
            if honeybee_location[1] == len(self.env[0])-1:
                nextposition.append(L)
            elif honeybee_location[1] == 0:
                nextposition.append(R)
            else:
                nextposition.extend([L,R])

        elif honeybee_location[0] in range(1,len(self.env)) and honeybee_location[1] == 0:
            nextposition.extend([R, U, D])

        elif honeybee_location[0] in range(1,len(self.env)) and honeybee_location[1] == len(self.env[0]):
            nextposition.extend([R, U, D])

        else:
            nextposition.extend([U, D, L, R])

        nextDirection = random.choice(nextposition)

        newlocation = tuple()

        if U == nextDirection:                
            newlocation= (honeybee_location[0]-1, honeybee_location[1])
        elif D == nextDirection:
            if honeybee_location[0] == 4:
                print(honeybee_location[0])
                newlocation = (honeybee_location[0], honeybee_location[1])
            else:
                newlocation = (honeybee_location[0]+1, honeybee_location[1])
        elif R == nextDirection:
            if honeybee_location[1] == 4:
                newlocation = (honeybee_location[0], honeybee_location[1])
            else:
                newlocation = (honeybee_location[0], honeybee_location[1]+1)            
        else:
            newlocation = (honeybee_location[0],honeybee_location[1]-1)

        return newlocation

    def perceptsin(self, honeybee_location):           
            nextstate = self.env[honeybee_location[0]][honeybee_location[1]].ObjP()    
            return nextstate
        

    def matchrules(self, currentstate):
        rules = [("flower", "suck"),("cactus", "Die")]  
    
        
        action = ""
        if currentstate in [cactus]:
            self.alive = True
            action = currentstate
            model[currentstate] = "Die"
        elif currentstate in [flower]: 
            self.alive = True
            action = "suck"
            model[currentstate] = "suck"            
        else:
            action = currentstate

    def program(self, goal_location):
        noofsteps = 0
        performance = 0

        while self.honeybee_location[0] != goal_location[0] and self.honeybee_location[1] != goal_location[1]:
            next_location = self.nextDirection(self.honeybee_location)
            self.honeybee_location = next_location

            if next_location[0] == self.goal_location[0]-1 and next_location[1] == self.goal_location[1]-1:
                self.isGoalAchieved = True
                print("HoneyBee suck the nectar from goldenflower at (",next_location[0]+1,", ",next_location[1]+1 ,") : Goal Achived") 
                performance += 100
                print("Total step performed : ", noofsteps)
                print("Performance : ", performance)
                return
            
            print("HoneyBee flied to ", next_location)
             
            currentstate = self.perceptsin(next_location)
            action = self.matchrules(currentstate)

            if self.alive == False:
                print("HoneyBee find the cactus : ", currentstate) 
                performance -= 20
            elif currentstate in [flower]:
                print("HoneyBee suck the nectar from flower : ","suck")
                performance += 50
            else:
                print("Nothing present in the location")
                performance -= 10
            noofsteps += 1
 

        print("Total step performed : ", noofsteps)
        print("Performance : ", performance)

print("**************************** Goal and model based Agent ********************************")

objmodelgoal = HoneyBeeModelBasedAgent(ENV, [0,0], [5,5])
objmodelgoal.program([5,5])
print("Maintain percept history : " ,model)


print("****************************************************************************************")


Garden_map = UndirectedGraph(dict(
    Agent=dict(none0_1=100, none1_0=400),
    none0_1=dict(Agent=200, cactus1_1=300, none0_2=500),
    none0_2=dict(none0_1=100, cactus0_3=700, flower1_2=600),
    cactus0_3=dict(none0_2=120, none1_3 = 150,flower0_4 = 100),
    flower0_4=dict(cactus0_3=170 ,cactus1_4=450),
    none1_0=dict(Agent =400 ,cactus1_1 =750,flower2_0=160),
    cactus1_1=dict(none1_0= 200,cactus2_1 = 340, flower1_2 = 160),
    flower1_2=dict(cactus1_1=180, none2_2=144,none1_3 =154),
    none1_3=dict(flower1_2=569, cactus1_4=607,none2_3=390),
    cactus1_4=dict(none1_3=789, none2_4=800, flower0_4 = 900),
    flower2_0=dict(none1_0=560,none3_0=456,cactus2_1=67),
    cactus2_1=dict(flower2_0=10,flower3_1=122,none2_2=216,cactus1_1=12),
    none2_2=dict(cactus2_1=89,flower1_2=567,cactus3_2=890,none2_3=45,),
    none2_3=dict(none2_2=27,none1_3=73,flower3_3=789,none2_4=67),
    none2_4=dict(none2_3=78,cactus1_4 =190,none3_4=89),
    none3_0=dict(flower2_0=145,none4_0=178,flower3_1=15),
    flower3_1= dict(none3_0 =567 ,cactus2_1=456,none4_1=78,cactus3_2=98),
    cactus3_2 = dict(flower3_1=67,none4_2=89,none2_2=788,flower3_3=56),
    flower3_3 = dict(cactus3_2=16,none4_3=145,none2_3=199,none3_4=178),
    none3_4 = dict(flower3_3=189,none2_4=79,GoldenFlower4_4=12),
    none4_0 = dict(none3_0=67,none4_1=78),
    none4_1 = dict(none4_0=89,flower3_1=199,none4_2=56),
    none4_2 = dict(none4_1=670,cactus3_2=90,none4_3=700),
    none4_3 = dict(none4_2=517,flower3_3=619,GoldenFlower4_4=85),
    GoldenFlower4_4 = dict(none4_3=89,none3_4=199)))

Garden_map.locations = dict(
    Agent=(0, 0), none0_1=(0, 1),none0_2=(0, 2), cactus0_3=(0, 3),flower0_4=(0, 4),
    none1_0=(1, 0), cactus1_1=(1, 1), flower1_2=(1, 2),none1_3 = (1,3),cactus1_4=(1,4),
    flower2_0=(2, 0), cactus2_1=(2, 1),none2_2=(2,2), none2_3=(2, 3),none2_4=(2,4),
    none3_0=(3, 0), flower3_1=(3, 1), cactus3_2=(3, 2),flower3_3 =(3,3),none3_4 =(3,4),
    none4_0=(4, 0), none4_1=(4, 1), none4_2=(4, 2),none4_3= (4,3),GoldenFlower4_4 =(4,4))


print("****************************************************************************************")


print("                                SEARCH ALGORITHM                                ")
print("****************************************************************************************")
print("                                UNINFORMED SEARCH ALGORITHM                                ")
print("****************************************************************************************")



print(" BREADTH FIRST SEARCH")
Goldenflower_problem_BFS = GraphProblem('Agent', 'GoldenFlower4_4', Garden_map)
result_BFS = breadth_first_tree_search(Goldenflower_problem_BFS)
print(" BREADTH FIRST SEARCH PATH COST : " , result_BFS.path_cost)
print(" BREADTH FIRST SEARCH COST :" ,result_BFS.solution())

print("****************************************************************************************")
print("****************************************************************************************")
print(" DEPTH-FIRST GRAPH SEARCH")



Goldenflower_problem_DFS = GraphProblem('Agent', 'GoldenFlower4_4', Garden_map)
result_DFS = depth_first_graph_search(Goldenflower_problem_DFS)
print(" DEPTH-FIRST GRAPH SEARCH PATH COST : " , result_DFS.path_cost)
print(" DEPTH-FIRST GRAPH SEARCH PATH : " ,result_DFS.solution())


print("****************************************************************************************")
print("****************************************************************************************")
print(" UNIFORM COST SEARCH ")



Goldenflower_problem_UCS = GraphProblem('Agent', 'GoldenFlower4_4', Garden_map)
result_UCS = uniform_cost_search(Goldenflower_problem_UCS)
print(" UNIFORM COST SEARCH PATH COST : " , result_UCS.path_cost)
print(" UNIFORM COST SEARCH PATH : " ,result_UCS.solution())


print("****************************************************************************************")
print("****************************************************************************************")

print("                                INFORMED SEARCH ALGORITHM                                ")
print("****************************************************************************************")

print(" BEST FIRST SEARCH")



Goldenflower_problem_BEST_FIRST_SEARCH = GraphProblem('Agent', 'GoldenFlower4_4', Garden_map)
result_BEST_FIRST_SEARCH= best_first_graph_search(Goldenflower_problem_BEST_FIRST_SEARCH, lambda node: node.state)
print(" BEST FIRST SEARCH PATH COST : " , result_BEST_FIRST_SEARCH.path_cost)
print(" BEST FIRST SEARCH PATH : " ,result_BEST_FIRST_SEARCH.solution())


print("****************************************************************************************")
print("****************************************************************************************")

print(" A* SEARCH")


Goldenflower_problem_A = GraphProblem('Agent', 'GoldenFlower4_4', Garden_map)
result_A= astar_search(Goldenflower_problem_A)
print(" A* SEARCH PATH COST : " , result_A.path_cost)
print(" A* SEARCH PATH : " ,result_A.solution())

print("****************************************************************************************")
print("****************************************************************************************")

print(" RECURSIVE BEST FIRST SEARCH")

Goldenflower_problem_R = GraphProblem('Agent', 'GoldenFlower4_4', Garden_map)
result_R= recursive_best_first_search(Goldenflower_problem_R)
print(" RECURSIVE BEST FIRST SEARCH PATH COST : " , result_R.path_cost)
print(" RECURSIVE BEST FIRST SEARCH PATH : " ,result_R.solution())


