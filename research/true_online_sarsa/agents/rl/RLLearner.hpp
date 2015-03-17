/****************************************************************************************
** Abstract class that needs to be implemented by any ALE agent. It defines the method 
** that controls the agent, forcing it to act. By using it we ensure that any approach,
** based in RL can be easily run, following the same pattern.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef AGENT_H
#define AGENT_H
#include "../../../../src/agents/Agent.hpp"
#endif

class RLLearner : public Agent{
	protected:
		ActionVect actions;

		double gamma, epsilon;
		double firstReward;
		bool   sawFirstReward;

		int    frame, toUseOnlyRewardSign, toBeOptimistic;
		int    randomActionTaken, numActions, numStepsPerAction;
		int    episodeLength, numEpisodesLearn, numEpisodesEval;

		/**
 		* It acts in the environment and makes the proper operations in the reward signal (normalizing,
 		* being optimistic, etc).
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		* @param int action action to be taken
 		*
 		* @param vector<double>& reward this vector is used to return, by reference, the reward observed
 		* by executing the action defined as parameter. This vector has two positions: in the first position
 		* the reward to be used by the RL algorithm is returned; in the second position, the game score is
 		* returned.
 		*/
		void act(ALEInterface& ale, int action, vector<double> &reward);

		/**
 		* Implementation of an epsilon-greedy function. Epsilon is defined in the constructor,
 		* in the argument Parameters *param
 		*
 		* @return int action to be taken
 		*/
		int epsilonGreedy(vector<double> &QValues);

		/**
		* Constructor to be used by the RL classes to save the parameters that
		* will be used by other methods.
		*
		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		* @param Parameters *param object containing the parameters passed to the algorithm, both by
 		*        file and command line.
 		*
		*/
		RLLearner(ALEInterface& ale, Parameters *param);

	public:
	   /**
 		* Pure virtual method that needs to be implemented by any RL agent.
 		*
 		* TODO: it may be useful to return something for the caller, as the total reward or policy. 
 		*       Additionally, it may be important to persist what was learned, maybe with another
 		*       pure virtual method?
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		* @param Features *features object that defines what feature function that will be used by the RL
 		*        agents.
 		*/
		virtual void learnPolicy(ALEInterface& ale, Features *features) = 0;

		/**
 		* Pure virtual method that needs to be implemented by any agent. Once the agent learned a
 		* policy it executes this policy for a given number of episodes. The policy is stored in
 		* the class' object.
 		*
 		* TODO it may be useful to return something for the caller, as an indicator of performance. 
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		* @param Features *features object that defines what feature function that will be used by the RL
 		*        agents. It may be null for other approaches as in Planning.
 		*/
		virtual void evaluatePolicy(ALEInterface& ale, Features *features) = 0;

		/**
		* Destructor, not necessary in this class.
		*/
		virtual ~RLLearner(){};
};
