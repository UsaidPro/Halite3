# Halite3
OpenAI Gym implementation of the Halite III game, to make training RL algorithms easier

This Gym environment _does not_ depend on the Halite III environment (it is a complete standalone). Instead of using Halite III's input/output-stream based engine this environment uses a custom engine I created. It isn't a perfect clone of Halite III's engine but pretty close. Since it doesn't depend on input/output streams, it doens't have the delay associated with actions that Halite III's engine does. This should __hopefully__ mean that training using this environment is fast.

The environment is stored on Halite3/haliteenv. I plan to clean up this repository and make it easier to use (this is on Github mostly for personal use, but public in case others might benefit).
