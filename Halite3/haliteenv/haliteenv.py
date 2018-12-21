import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import gym

class HaliteEnv(gym.Env):
   metadata = {'render_modes':['human']}
   def __init__(self, numPlayers, mapType, mapSize):
      print("Initializing Halite Environment")
      self.map = Map.generateFractalMap(mapSize.value, numPlayers)
   
   def render(self, mode = 'human'):
      fig = plt.figure(figsize=(8, 8))
      fig.add_subplot(1, 3, 1)
      plt.imshow(self.map[:, :, 0], cmap='hot', interpolation='nearest')
      fig.add_subplot(1, 3, 2)
      plt.imshow(self.map[:, :, 1], cmap='hot', interpolation='nearest')
      fig.add_subplot(1, 3, 3)
      plt.imshow(self.map[:, :, 2], cmap='hot', interpolation='nearest')
      plt.show()

class MapType(Enum):
   BASIC = 0
   FRACTAL = 1
   BLUR = 2
class MapSize(Enum):
   TINY = 32
   SMALL = 40
   MEDIUM = 48
   LARGE = 56
   GIANT = 64

class Map:
   def generateBasicMap(mapSize):
      #Default halite value is 10 before transformations
      map = np.empty((mapSize, mapSize, 3))
      map[:, :, 0].fill(10)
      return map
   
   def generateBlurMap(mapSize):
      print("Blur")
   
   def generateSmoothNoise(sourceNoise, wavelength):
      miniSource = np.zeros((np.ceil(float(sourceNoise.shape[0]) / wavelength).astype(np.int64), np.ceil(float(sourceNoise.shape[1]) / wavelength).astype(np.int64)))
      for y in range(0, miniSource.shape[0]):
         for x in range(0, miniSource.shape[1]):
            miniSource[y, x] = sourceNoise[wavelength * y, wavelength * x]
      smoothedSource = np.zeros_like(sourceNoise)
      for y in range(0, sourceNoise.shape[0]):
         yI = int(y / wavelength)
         yF = int(y / wavelength + 1) % miniSource.shape[0]
         verticalBlend = float(y) / wavelength - yI
         for x in range(0, sourceNoise.shape[1]):
            xI = int(x / wavelength)
            xF = int(x / wavelength + 1) % miniSource.shape[1]
            horizontalBlend = float(x) / wavelength - xI
            topBlend = (1 - horizontalBlend) * miniSource[yI][xI] + horizontalBlend * miniSource[yI][xF]
            bottomBlend = (1 - horizontalBlend) * miniSource[yF][xI] + horizontalBlend * miniSource[yF][xF]
            smoothedSource[y][x] = (1 - verticalBlend) * topBlend + verticalBlend * bottomBlend
      return smoothedSource

   def generateFractalMap(mapSize, numPlayers):
      numTiles = 1
      numTileRows = 1
      numTileCols = 1
      while numTiles < numPlayers:
         numTileCols *= 2
         numTiles *= 2
         if numTiles is numPlayers:
            break
         numTileRows *= 2
         numTiles *= 2
      tileWidth = int(mapSize / numTileCols)
      tileHeight = int(mapSize / numTileRows)
      sourceNoise = np.square(np.random.uniform(0.0, 1.0, (tileHeight, tileWidth)))
      region = np.zeros((tileHeight, tileWidth))
      maxOctave = np.floor(np.log2(min(tileHeight, tileWidth))) + 1
      amplitude = 1.0
      for octave in np.arange(2, maxOctave + 1, 1):# range(2, maxOctave + 1):
         smoothedSource = Map.generateSmoothNoise(sourceNoise, int(round(pow(2, maxOctave - octave))))
         region += amplitude * smoothedSource
         amplitude *= Constants.PERSISTENCE
      amplitude += amplitude * smoothedSource
      region = np.square(region)
      maxCellProduction = np.random.randint(0, 7296) % (1 + Constants.MAX_CELL_PRODUCTION - Constants.MIN_CELL_PRODUCTION) + Constants.MIN_CELL_PRODUCTION
      region *= maxCellProduction / region.max()
      tile = np.empty((tileHeight, tileWidth, 3))
      tile[:, :, 0] = np.round(region)
      tile[:, :, 1].fill(0)
      tile[:, :, 2].fill(0)
      factoryX = int(tileWidth / 2)
      factoryY = int(tileHeight / 2)
      if tileWidth >= 16 and tileWidth <= 40 and tileHeight >= 16 and tileHeight <= 40:
         factoryX = 8 + ((tileWidth - 16) / 24.0) * 20
         if numPlayers > 2:
            factoryY = 8 + ((tileHeight - 16) / 24.0) * 20
      tile[factoryY, factoryX, 1] = 1
      tile[factoryY, factoryX, 2] = 1
      currentWidth = tileWidth
      currentHeight = tileHeight
      numTiles = 1
      while numTiles < numPlayers:
         #Flipping over vertical line
         flip = np.fliplr(tile)
         tile = np.concatenate((tile, flip), axis=1)
         currentWidth *= 2
         numTiles *= 2
         if numTiles is numPlayers:
            break
         #Flipping over horizontal line
         flip = np.flipud(tile)
         tile = np.concatenate(tile, flip)
         currentHeight *= 2
         numTiles *= 2
      playerNum = 1
      for i in range(1, int(currentWidth / tileWidth) + 1):
         for j in range(1, int(currentHeight / tileHeight) + 1):
            tile[((j - 1) * tileHeight):(j * tileHeight), ((i - 1) * tileWidth):(i * tileWidth), 2] *= playerNum
            playerNum += 1
      return tile
   
class Constants:
   CAPTURE_ENABLED = False
   CAPTURE_RADIUS = 3
   DEFAULT_MAP_HEIGHT = 48
   DEFAULT_MAP_WIDTH = 48
   DROPOFF_COST = 4000
   DROPOFF_PENALTY_RATIO = 4
   EXTRACT_RATIO = 4
   FACTOR_EXP_1 = 2.0
   FACTOR_EXP_2 = 2.0
   INITIAL_ENERGY = 5000
   INSPIRATION_ENABLED = True
   INSPIRATION_RADIUS = 4
   INSPIRATION_SHIP_COUNT = 2
   INSPIRED_BONUS_MULTIPLIER = 2.0
   INSPIRED_EXTRACT_RATIO = 4
   INSPIRED_MOVE_COST_RATIO = 10
   MAX_CELL_PRODUCTION = 1000
   MAX_ENERGY = 1000
   MAX_PLAYERS = 16
   MAX_TURNS = 500
   MAX_TURN_THRESHOLD = 64
   MIN_CELL_PRODUCTION = 900
   MIN_TURNS = 400
   MIN_TURN_THRESHOLD = 32
   MOVE_COST_RATIO = 10
   NEW_ENTITY_ENERGY_COST = 1000
   PERSISTENCE = 0.7
   SHIPS_ABOVE_FOR_CAPTURE = 3
   STRICT_ERRORS = False