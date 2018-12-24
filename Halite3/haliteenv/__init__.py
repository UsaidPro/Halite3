from gym.envs.registration import register
from haliteenv.haliteenv import HaliteEnv, Constants

register(
   id='HEnv2PTrain-v0',
   entry_point='haliteenv.haliteenv:HaliteEnv',
   kwargs={
      'numPlayers':2,
      'mapType':haliteenv.MapType.BASIC,
      'mapSize':haliteenv.MapSize.MEDIUM,
      'regenMapOnReset':False
   },
)
register(
   id='HEnv2PTrain-v1',
   entry_point='haliteenv.haliteenv:HaliteEnv',
   kwargs={
      'numPlayers':2,
      'mapType':haliteenv.MapType.BASIC,
      'mapSize':haliteenv.MapSize.MEDIUM,
      'regenMapOnReset':True
   },
)