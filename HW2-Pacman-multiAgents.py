#Melih Şahin Comp 341 HW2

# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        current_evaluation=0
        ghost_positions=successorGameState.getGhostPositions()

        old_scared_times=[ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
        def _distance(x,y):
            sum=0
            for c in range(2):
                sum+=abs(x[c]-y[c])
            return sum

        for a in range(len(ghost_positions)):
           if ghost_positions[a]==successorGameState.getPacmanPosition():
               current_evaluation+=-1000
        if (len(newFood.asList())<len(currentGameState.getFood().asList())):
            current_evaluation+=100
        if (sum(newScaredTimes)>=sum(old_scared_times)):
            current_evaluation+=500

        new_total_distance=0
        for j in range(len(newFood.asList())):
            new_total_distance+=_distance(successorGameState.getPacmanPosition(),newFood.asList()[j])

        previous_total_distance=0
        for i in range(len(currentGameState.getFood().asList())):
            previous_total_distance+=_distance(currentGameState.getPacmanPosition(),currentGameState.getFood().asList()[i])
        current_evaluation+=(previous_total_distance-new_total_distance)

        if action=="Stop":
            current_evaluation+=-10
        #print(current_evaluation)
        return current_evaluation
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"



        def max_chooser(current_depth, game_state, depth):
            possible_states_scores=[]
            possible_actions=game_state.getLegalActions(0)



            for a in range(len(possible_actions)):

                new_state=game_state.generateSuccessor(0,possible_actions[a])
                if new_state.isWin() or new_state.isLose():
                    possible_states_scores.append(scoreEvaluationFunction(new_state))
                else:
                    if current_depth<=depth :
                        number1 = new_state.getNumAgents()
                        possible_states_scores.append(min_chooser(current_depth, new_state, depth, 1, number1))
                    else:
                        possible_states_scores.append(scoreEvaluationFunction(new_state))
            return max(possible_states_scores), possible_actions[possible_states_scores.index(max(possible_states_scores))]


        def min_chooser(current_depth, game_state, depth, current_layer, total_iteration_number):
            possible_states_scores = []


            possible_ghost_actions = game_state.getLegalActions(current_layer)
            for a in range(len(possible_ghost_actions)):
                new_state = game_state.generateSuccessor(current_layer, possible_ghost_actions[a])
                if new_state.isWin() or new_state.isLose():
                    possible_states_scores.append(scoreEvaluationFunction(new_state))
                elif current_layer<total_iteration_number-1:
                    possible_states_scores.append(min_chooser(current_depth, new_state, depth, current_layer+1,total_iteration_number ))
                else:
                    if current_depth<depth:
                        k,l=max_chooser(current_depth+1, new_state, depth )
                        possible_states_scores.append(k)
                    else:
                        possible_states_scores.append(scoreEvaluationFunction(new_state))
            return min(possible_states_scores)

        def minimax_search(gameState, depth):
            a,b=max_chooser(1,gameState,depth)
            return b

        return minimax_search(gameState,self.depth)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_chooser(current_depth, game_state, depth, current_min=9999999999999999, parent_current_max=-9999999999999999):
            current_max=parent_current_max
            possible_states_scores=[]
            possible_actions=game_state.getLegalActions(0)




            for a in range(len(possible_actions)):

                if current_max>current_min:
                    return current_max, None


                new_state=game_state.generateSuccessor(0,possible_actions[a])
                if new_state.isWin() or new_state.isLose():
                    q1=scoreEvaluationFunction(new_state)
                    possible_states_scores.append(q1)
                    if q1>current_max:
                        current_max=q1

                else:
                    if current_depth<=depth :
                        number1 = new_state.getNumAgents()
                        q2=min_chooser(current_depth, new_state, depth, 1, number1,current_min, current_max)
                        if q2!=None:
                            possible_states_scores.append(q2)
                            if q2 > current_max:
                                current_max = q2

                    else:
                        q3=scoreEvaluationFunction(new_state)
                        possible_states_scores.append(q3)
                        if q3 > current_max:
                            current_max = q3
            if len(possible_states_scores)==0:

                return current_max, None
            return max(possible_states_scores), possible_actions[possible_states_scores.index(max(possible_states_scores))]


        def min_chooser(current_depth, game_state, depth, current_layer, total_iteration_number , parent_current_min, current_max):

            current_min=parent_current_min
            possible_states_scores = []


            possible_ghost_actions = game_state.getLegalActions(current_layer)
            for a in range(len(possible_ghost_actions)):


                if current_min<current_max:
                    return current_min

                new_state = game_state.generateSuccessor(current_layer, possible_ghost_actions[a])
                if new_state.isWin() or new_state.isLose():
                    q1=scoreEvaluationFunction(new_state)
                    possible_states_scores.append(q1)
                    if q1<current_min:
                        current_min=q1
                elif current_layer<total_iteration_number-1:
                    q2=min_chooser(current_depth, new_state, depth, current_layer+1,total_iteration_number, current_min,current_max)

                    possible_states_scores.append(q2)
                    if q2<current_min:
                        current_min=q2
                else:
                    if current_depth<depth:
                        k,l=max_chooser(current_depth+1, new_state, depth, current_min, current_max)
                        if k!=None:
                            possible_states_scores.append(k)
                            if k < current_min:
                                current_min = k

                    else:
                        q4=scoreEvaluationFunction(new_state)
                        possible_states_scores.append(q4)
                        if q4 < current_min:
                            current_min = q4

            if len(possible_states_scores)==0:

                return current_min
            return min(possible_states_scores)

        def minimax_search(gameState, depth):
            a,b=max_chooser(1,gameState,depth)
            return b

        return minimax_search(gameState,self.depth)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_chooser(current_depth, game_state, depth):
            possible_states_scores=[]
            possible_actions=game_state.getLegalActions(0)



            for a in range(len(possible_actions)):

                new_state=game_state.generateSuccessor(0,possible_actions[a])
                if new_state.isWin() or new_state.isLose():
                    possible_states_scores.append(scoreEvaluationFunction(new_state))
                else:
                    if current_depth<=depth :
                        number1 = new_state.getNumAgents()
                        possible_states_scores.append(min_chooser(current_depth, new_state, depth, 1, number1))
                    else:
                        possible_states_scores.append(scoreEvaluationFunction(new_state))
            return max(possible_states_scores), possible_actions[possible_states_scores.index(max(possible_states_scores))]


        def min_chooser(current_depth, game_state, depth, current_layer, total_iteration_number):
            possible_states_scores = []


            possible_ghost_actions = game_state.getLegalActions(current_layer)
            for a in range(len(possible_ghost_actions)):
                new_state = game_state.generateSuccessor(current_layer, possible_ghost_actions[a])
                if new_state.isWin() or new_state.isLose():
                    possible_states_scores.append(scoreEvaluationFunction(new_state))
                elif current_layer<total_iteration_number-1:
                    possible_states_scores.append(min_chooser(current_depth, new_state, depth, current_layer+1,total_iteration_number ))
                else:
                    if current_depth<depth:
                        k,l=max_chooser(current_depth+1, new_state, depth )
                        possible_states_scores.append(k)
                    else:
                        possible_states_scores.append(scoreEvaluationFunction(new_state))
            return sum(possible_states_scores)/len(possible_states_scores)

        def minimax_search(gameState, depth):
            a,b=max_chooser(1,gameState,depth)
            return b

        return minimax_search(gameState,self.depth)

def betterEvaluationFunction(currentGameState):

    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    print("******")
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newcapsules = currentGameState.getCapsules()



    current_evaluation = 0
    ghost_positions = currentGameState.getGhostPositions()

    def _distance(x, y):
        sum = 0
        for c in range(2):
            sum += abs(x[c] - y[c])
        return sum

    min_ghost_distance=1000
    for a in range(len(ghost_positions)):
        min_ghost_distance=min(_distance(ghost_positions[a],newPos),min_ghost_distance)
        if newPos==ghost_positions[a]:
            current_evaluation-=100000
    current_evaluation+=min_ghost_distance


    min_food=10000
    for b in range(len(newFood)):
        min_food=min(min_food,_distance(newFood[b],newPos))
    current_evaluation-=len(newFood)*100-min_food*10

    current_evaluation+=sum(newScaredTimes)


    min_capsule=10000

    for c in range(len(newcapsules)):
        min_capsule=min(min_capsule,_distance(newcapsules[b],newPos))
    current_evaluation-=min_capsule*10+len(newcapsules)



    current_evaluation+=currentGameState.getScore()

    return current_evaluation


# Abbreviation
better = betterEvaluationFunction
