import numpy as np
import gym
import haliteenv
from haliteenv import Constants
import time

halite = gym.make('HEnv2PTrain-v0')
mapShape = (halite.metadata['map_size'], halite.metadata['map_size'], halite.metadata['num_players'])

startTime = time.clock()
action = np.ones(mapShape)
mapObs, reward = halite.step(action)
NUM_STEPS = 5000
dropoffs = 0
for i in range(0, NUM_STEPS):
   action = np.zeros(mapShape, np.int64)
   for player in range(0, halite.metadata['num_players']):
      if i % 2 == 0:
         ships = np.where(mapObs[0][:, :, 3] == 1)
         for loc in range(0, len(ships[0])):
            if((mapObs[1][player] + mapObs[0][ships[0][loc], ships[1][loc], 0]) > Constants.DROPOFF_COST and mapObs[0][ships[0][loc], ships[1][loc], 4] == player + 1 and mapObs[0][ships[0][loc], ships[1][loc], 2] == 0 and dropoffs < 4):
               action[ships[0][loc], ships[1][loc], player] = 2 #Convert to dropoff
               dropoffs += 1
            else:
               action[ships[0][loc], ships[1][loc], player] = np.random.randint(3, 7)#Do random action
      else:
         factories = np.where(mapObs[0][:, :, 2] == 1)
         for loc in range(0, len(factories[0])):   
            #Spawn ships whenever possible
            #The environment would just ignore if no ship can be spawned
            action[factories[0][loc], factories[1][loc], player] = 1
   mapObs, reward = halite.step(action)
timeTaken = time.clock() - startTime
print(str(timeTaken) + " seconds")
print("Average time per loop: " + str(timeTaken / NUM_STEPS))
print("Finished! Rendering . . .")
halite.render()