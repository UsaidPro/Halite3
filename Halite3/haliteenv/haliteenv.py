import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import gym

class HaliteEnv(gym.Env):
   """
   Stores the Halite III OpenAI gym environment.
   This environment does not use Halite III's actual game engine 
   (which analyzes input from terminal and is slow for RL) but instead is
   a replica in Python.
   
   Attributes:
   -----------
   self.map : np.ndarray
      Map of game as a 3D array. Stores different information on each "layer"
      of the array.
      Layer 0: The Halite currently on the sea floor
      Layer 1: The Halite currently on ships/factory/dropoff
      Layer 2: Whether a Factory or Dropoff exists at the layer (Factory is 1, Dropoff is -1)
      Layer 3: Whether a Ship exists at the layer
      Layer 4: Ownership
      Layer 5: Inspiration (not given as part of observation by default)
   
   self.playerHalite : np.ndarray
      Stores the total halite a player with ownership id <index + 1> has. self.map also stores the total halite 
      with the halite under factories/dropoffs, but doesn't include the 5000 initial.
   """
   metadata = {'render_modes':['human']}
   
   def __init__(self, numPlayers, mapType, mapSize):
      """
      HaliteEnv initialization function.
      """
      print("Initializing Halite Environment")
      self.map = Map.generateFractalMap(mapSize.value, numPlayers)
      
      self.playerHalite = np.empty((numPlayers, 1))
      self.playerHalite.fill(5000)
      self.numPlayers = numPlayers
      self.mapSize = mapSize.value
   
   def step(self, action):
      """
      Step of Halite III environment
      
      Parameters:
      -----------
      action : array
         Array of length <numPlayers> where each element is an action for the player
         whose id is (index + 1)
         Each player's actions are represented as a 2D array the size of the map, where
         each element is what move to do on an element of the map. This means that many extra/
         illegal moves are made, which are ignored during the step.
         
         Why 2D? I couldn't think of a different way to represent the full action space.
         I'm open for suggestions (create an issue on the Github if you have any).
         
         Moves possible:
            0 - Do Nothing
            1 - Spawn ship
            2 - Convert to dropoff
            3 - Move N
            4 - Move E
            5 - Move S
            6 - Move W
      
      Returns:
      --------
      ob, reward, episode_over, info : tuple
         ob (array):
            Game observation as an sequence (<map>, <playerHalite>)
         reward (array):
            Reward for each player with ownership id <index + 1> for action taken
         episode_over (bool)
            Whether or not game is over.
         info (dict)
            Info for debugging. 
      """
      playerReward = np.zeros((numPlayers, 1))
      #Process turns first
      #Loop through ships and factories and pull from corresponding player's actions for that ship/factory
      ships = np.where(self.map[:, :, 3] == 1)
      for ship in ships:
         act = action[self.map[ship[0], ship[1], 4] - 1][ship[0]][ship[1]]
         success = True
         if(act == 2):
            success = self.constructDropoff(ship[1], ship[0])
         elif(act == 3):
            success = self.moveShip(ship[1], ship[0], 'N')
         elif(act == 4):
            success = self.moveShip(ship[1], ship[0], 'E')
         elif(act == 5):
            success = self.moveShip(ship[1], ship[0], 'S')
         elif(act == 6):
            success = self.moveShip(ship[1], ship[0], 'W')
         
         #Deinceventize outright bad/invalid moves
         if(not success):
            playerReward[self.map[ship[1], ship[0], 4] - 1] -= 0.1
      factories = np.where(self.map[:, :, 2] == 1)
      for factory in factories:
         act = action[self.map[factory[0], factory[1], 4] - 1][factory[0]][factory[1]]
         if(act == 1):
            success = self.spawnShip(factory[1], factory[0])
         #Deinceventize outright bad/invalid moves
         if(not success):
            playerReward[self.map[ship[1], ship[0], 4] - 1] -= 0.1

      #Reset inspiration map
      self.map[:, :, 5].fill(0)
      #Update inspiration/extraction
      nearShips = np.where(self.map[:, :, 3] == 1)
      maxEnergy = Constants.MAX_ENERGY
      bonusMultiplier = Constants.INSPIRED_BONUS_MULTIPLIER
      for ship in nearShips:
         #Remember ship[0] is y and ship[1] is x
         inspired = self.isInspired(ship[1], ship[0])
         ratio = Constants.INSPIRED_EXTRACT_RATIO if inspired else Constants.EXTRACT_RATIO
         extracted = np.ceil(self.map[ship[0], ship[1], 0] / ratio).astype(np.int64)
         gained = extracted
         if(extracted == 0 and self.map[ship[0], ship[1], 0] > 0):
            extracted = gained = self.map[ship[0], ship[1], 0]
         if(extracted + self.map[ship[0], ship[1], 1] > maxEnergy):
            extracted = maxEnergy - self.map[ship[0], ship[1], 1]
         if(inspired):
            gained += bonusMultiplier * gained
         if(maxEnergy - self.map[ship[0], ship[1], 1] < gained):
            gained = maxEnergy - self.map[ship[0], ship[1], 1]
         self.map[ship[0], ship[1], 1] += gained
         self.map[ship[0], ship[1], 0] -= extracted
         
         #Capture is currently disabled according to constants, so not adding it
      for playerId in range(0, len(playerReward)):
         playerReward[playerId] += playerHalite[playerId] * 0.0005
      
      return (self.map[:, :, :5], playerReward)
   
   def render(self, mode = 'human'):
      """
      Renders the current Halite III game environment as three plots for easier debugging. 
      The leftmost subplot is the current halite distribution on the map.
      The middle subplot is whether nothing/ship/factory exists at a location. 
      The rightmost subplot describes ownership.
      """
      fig = plt.figure(figsize=(8, 8))
      fig.add_subplot(2, 3, 1)
      plt.imshow(self.map[:, :, 0], cmap='hot', interpolation='nearest')
      fig.add_subplot(2, 3, 2)
      plt.imshow(self.map[:, :, 1], cmap='hot', interpolation='nearest')
      fig.add_subplot(2, 3, 3)
      plt.imshow(self.map[:, :, 2], cmap='hot', interpolation='nearest')
      fig.add_subplot(2, 3, 4)
      plt.imshow(self.map[:, :, 3], cmap='hot', interpolation='nearest')
      fig.add_subplot(2, 3, 5)
      plt.imshow(self.map[:, :, 4], cmap='hot', interpolation='nearest')
      fig.add_subplot(2, 3, 6)
      plt.imshow(self.map[:, :, 5], cmap='hot', interpolation='nearest')
      plt.show()

   def isInspired(self, shipX, shipY):
      """
      Determines if the ship at location is inspired.
      The calculation is not perfect - it missed inspired ships at far diagonals
      since it does calculations based on rectangles.
      TODO: Improve inspired calculations (add more rectangles to check)
      
      Parameters:
      -----------
      shipX : int
         X-coordinate of ship
      shipY : int
         Y-coordinate of ship
      
      Returns:
      --------
      result : bool
         Whether ship is inspired
      """
      #First check if ship is inspired - it would be inspired if it had a 1 at index 4
      if(self.map[shipY, shipX, 5] == 1):
         return true
      #If it isn't inspired check nearby ships
      #First rectangle check (13x11)
      marginX1 = shipX - 6
      if(marginX1 < 0):
         marginX1 = 0
      marginX2 = shipX + 6
      if(marginX2 >= self.map.shape[1]):
         marginX2 = self.map.shape[1] - 1
      marginY1 = shipY - 5
      if(marginX1 < 0):
         marginX1 = 0
      marginY2 = shipY + 5
      if(marginY2 >= self.map.shape[0]):
         marginY2 = self.map.shape[0] - 1
      #TODO: More rectangle checks to improve accuracy
      #TODO: Is there a NumPy improvement to below (so I don't have to compare against 1)?
      nearShips = np.where(self.map[marginY1:marginY2, marginX1:marginX2, 3] == 1)
      for ship in nearShips:
         x = ship[1] + marginX1
         y = ship[0] + marginY1
         if(self.map[y, x, 4] == self.map[shipY, shipX, 4]):
            #Found nearby ship, setting inspiration and returning
            self.map[marginY1:marginY2, marginX1:marginX2, 5] = 1
            return True
      self.map[marginY1:marginY2, marginX1:marginX2, 5] = 1
      return False

   def destroyShip(self, shipX, shipY):
      """
      Destroys ship at coordinates <X, Y>
      
      Parameters:
      -----------
      shipX : int
         X-coordinate of ship
      shipY : int
         Y-coordinate of ship
      """
      self.map[shipY, shipX, 0] += self.map[shipY, shipX, 1]
      self.map[shipY, shipX, 1] = 0
      self.map[shipY, shipX, 3] = 0
      if(self.map[shipY, shipX, 2] == 0):
         self.map[shipY, shipX, 4] = 0

   def constructDropoff(self, shipX, shipY):
      """
      Converts the Ship at <shipX, shipY> to a Dropoff.
      
      Parameters:
      -----------
      shipX : int
         X-coordinate of ship
      shipY : int
         Y-coordinate of ship

      Returns:
      --------
      bool
         Whether Dropoff construction was successful
      """
      #First, check if player has enough halite to construct dropoff. If not, return False
      if((self.playerHalite[self.map[shipY, shipX, 4] - 1] + self.map[shipY, shipX, 0] - Constants.DROPOFF_COST) < 0):
         return False
      self.playerHalite[self.map[shipY, shipX, 4] - 1] += self.map[shipY, shipX, 0]
      self.playerHalite[self.map[shipY, shipX, 4] - 1] -= Constants.DROPOFF_COST
      self.map[shipY, shipX, 3] = 0
      self.map[shipY, shipX, 2] = -1
      self.map[shipY, shipX, 0] = 0
      return True

   def moveShip(self, shipX, shipY, move):
      """
      Moves the ship at <shipX, shipY> in the direction specified by <move>
   
      Parameters:
      -----------
      shipX : int
         X-coordinate of ship
      shipY : int
         Y-coordinate of ship
      move : char
         Direction to move in ('N', 'E', 'S', 'W')
      
      Returns:
      --------
      bool
         Whether turn was successful
      """
      #First, check if ship has enough halite to move. If not, return False
      cost = Constants.INSPIRED_MOVE_COST_RATIO if self.map[shipY, shipX, 5] == 1 else Constants.MOVE_COST_RATIO
      required = self.map[shipY, shipX, 0] / cost
      if(self.map[shipY, shipX, 1] < required):
         return False
      if(move == 'N'):
         if(shipY == 0):
            #Cannot move North
            return False
         else:
            return self.attemptMove(shipX, shipY, shipX, shipY - 1)
      elif(move == 'E'):
         if(shipX == self.map.shape[1] - 1):
            #Cannot move East
            return False
         else:
            return self.attemptMove(shipX, shipY, shipX + 1, shipY)
      elif(move == 'S'):
         if(shipY == self.map.shape[0] - 1):
            #Cannot move South
            return False
         else:
            return self.attemptMove(shipX, shipY, shipX, shipY - 1)
      else:
         #Moving West
         if(shipX == 0):
            return False
         else:
            return self.attemptMove(shipX, shipY, shipX - 1, shipY)
   
   def attemptMove(self, shipX, shipY, newX, newY):
      """
      Helper function for moveShip(). Processes collisions and halite dropoff.

      Parameters:
      -----------
      shipX : int
         Origin X-coordinate of ship moving
      shipY : int
         Origin Y-coorinate of ship moving
      newX : int
         New X-coordinate of ship
      newY : int
         new Y-coordinate of ship
      
      Returns:
      --------
      bool
         Whether turn was successful
      """
      if(self.map[newY, newX, 3] == 1):
         #There exists a ship where we are trying to move
         if(self.map[shipY, shipX, 4] == self.map[newY, newX, 4]):
            #We are trying to hit our own ship.
            #Later, try removing this and see if some weird strategy emerges
            return False
         else:
            #Handle collisions
            #Note that having allied ships crash into each other is better at the end (to speed up dropoff)
            #I will need to rewrite this code if I decide to add hitting own ships
            self.map[shipY, shipX, 4] = 0
            self.map[shipY, shipX, 3] = 0
            self.map[newY, newX, 4] = 0
            self.map[newY, newX, 3] = 0
            self.map[newY, newX, 0] += self.map[shipY, shipX, 1] + self.map[newY, newX, 1]
            self.map[newY, newX, 1] = 0
            self.map[shipY, shipX, 1] = 0
            return True
      elif(np.abs(self.map[newY, newX, 2]) == 1):
         if(self.map[newY, newX, 4] == self.map[shipY, shipX, 4]):
            #Depositing into dropoff/factory
            self.map[newY, newX, 1] += self.map[shipY, shipX, 1]
            self.playerHalite[self.map[shipY, shipX, 4] - 1] += self.map[shipY, shipX, 1]
            #Removing ship's owned halite
            self.map[shipY, shipX, 1] = 0
            #Moving ship on top
            self.map[shipY, shipX, 4] = 0
            self.map[shipY, shipX, 3] = 0
            self.map[newY, newX, 3] = 1
            return True
         else:
            #We are on top of enemy factory! To make my spawnShip function easier, I'm going to deincentivize this
            return False
      else:
         #Just moving on
         self.map[newY, newX, 4] = self.map[shipY, shipX, 4]
         self.map[newY, newX, 3] = 1
         self.map[newY, newX, 1] = self.map[shipY, shipX, 1]
         self.map[shipY, shipX, 4] = 0
         self.map[shipY, shipX, 3] = 0
         self.map[shipY, shipX, 1] = 0
         return True
         
   def spawnShip(self, factoryX, factoryY):
      """
      Spawns a Ship at coordinates <factoryX, factoryY>

      Parameters:
      -----------
      factoryX : int
         X-coordinate of factory
      factoryY : int
         Y-coordinate of factory

      Returns:
      --------
      bool
         Whether spawning ship was succesful
      """
      if(self.playerHalite[self.map[factoryY, factoryX, 4] - 1] - Constants.NEW_ENTITY_ENERGY_COST < 0):
         return False
      else:
         if(self.map[factoryY, factoryX, 3] == 1):
            #Ship exists on top of factory already - don't create
            return False
         else:
            self.playerHalite[self.map[factoryY, factoryX, 4] - 1] -= Constants.NEW_ENTITY_ENERGY_COST
            self.map[factoryY, factoryX, 3] = 1
            self.map[factoryY, factoryX, 1] = 0
            return True

class MapType(Enum):
   """
   Enum of the different map types
   """
   BASIC = 0
   FRACTAL = 1
   BLUR = 2
class MapSize(Enum):
   """
   Enum of the different possible map sizes
   """
   TINY = 32
   SMALL = 40
   MEDIUM = 48
   LARGE = 56
   GIANT = 64

class Map:
   """
   Class that holds the map-generation functions
   """
   def generateBasicMap(mapSize):
      """
      Generates a basic map (a map with all cells having 10 halite)
      TODO: Complete function to include factories and players
      """
      #Default halite value is 10 before transformations
      map = np.empty((mapSize, mapSize, 3))
      map[:, :, 0].fill(10)
      return map
   
   def generateBlurMap(mapSize):
      """
      Stub of future generator for a blur-style map
      """
      print("Blur")
   
   def generateSmoothNoise(sourceNoise, wavelength):
      """
      Helper function for generateFractalMap. Generates smoothed noise for fractals
      """
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
      """
      Generates fractal-based map
      """
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
      tile = np.empty((tileHeight, tileWidth, 6))
      #Halite on floor
      tile[:, :, 0] = np.round(region)
      #Halite on ships
      tile[:, :, 1].fill(0)
      #Factories
      tile[:, :, 2].fill(0)
      #Ships
      tile[:, :, 3].fill(0)
      #Ownership
      tile[:, :, 4].fill(0)
      #Inspiration
      tile[:, :, 5].fill(0)
      
      factoryX = int(tileWidth / 2)
      factoryY = int(tileHeight / 2)
      if tileWidth >= 16 and tileWidth <= 40 and tileHeight >= 16 and tileHeight <= 40:
         factoryX = 8 + ((tileWidth - 16) / 24.0) * 20
         if numPlayers > 2:
            factoryY = 8 + ((tileHeight - 16) / 24.0) * 20
      tile[factoryY, factoryX, 0] = 0
      tile[factoryY, factoryX, 2] = 1
      tile[factoryY, factoryX, 4] = 1
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
            tile[((j - 1) * tileHeight):(j * tileHeight), ((i - 1) * tileWidth):(i * tileWidth), 4] *= playerNum
            playerNum += 1
      return tile

class Constants:
   """
   Stores the constants used by the Halite III game as of
   early December 2018
   """
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