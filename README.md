# ichack26

This branch contains a tidied, revised version of the script we used to test the main algorithm. There are two .json files which contain saved street data from london. The map can be drawn on, and there is live feedback on the pathway that the algorithm selects.

The algorithm is a variation of Dijkstra, inspired by A*. Instead of using a heuristic to affect ordering, the heuristic here is directly added to the distance, creating an artificial increase in distance when the selected route deviates from the drawing. This simple, elegant modification of a known algorithm can be seen to be extremely effective at achieving the desired goal.

To improve performance, a search (referenced in code as a connect), is only performed on a fixed number of drawing vertices. For points which are far out enough, we assume that the found route is unlikely to change. This value is configurable.
