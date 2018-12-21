from gym.envs.registration import register
from haliteenv.haliteenv import HaliteEnv

register(
   id='HEnv2Player-v0',
   entry_point='haliteenv.haliteenv:HaliteEnv',
   kwargs={
      'numPlayers':2,
      'mapType':haliteenv.MapType.BASIC,
      'mapSize':haliteenv.MapSize.MEDIUM,
   },
)