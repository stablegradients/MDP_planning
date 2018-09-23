import numpy as np
import pulp
import argparse

class MDP:
	def __init__(self,probability):
		self.probability=probability
		self.actions=100
		self.states=self.actions+1

		self.gambler_reward=np.zeros([self.states,self.actions,self.states])
		self.gambler_transition=np.zeros([self.states,self.actions,self.states])
		
		for s in range(self.states):
			for a in range(min(s,self.actions-s)):
				if s ==0 or s==self.actions:
					self.gambler_transition[s,:,s]=np.ones([self.actions])
				else:
					self.gambler_transition[s,a,s+a+1]=self.probability
					self.gambler_transition[s,a,s-a-1]=1-self.probability
				
				if s+a+1 == self.actions:
					self.gambler_reward[s,a,self.actions]=1
				
		self.gamma=1
		self.gambler_continuous=False
		return
	def dump_mdp(self):
		print(self.states)
		print(self.actions)

		for s in range(self.states):
			for a in range(self.actions):
				for s_ in range(self.states):
					print str(self.gambler_reward[s,a,s_]) + "\t",
				print "\n",
		for s in range(self.states):
			for a in range(self.actions):
				for s_ in range(self.states):
					print str(self.gambler_transition[s,a,s_]) + "\t",
				print "\n",
		print self.gamma 
		
		print "episodic"

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--probability")
	
	args=parser.parse_args()
	obj=MDP(probability=float(args.probability))
	obj.dump_mdp()
main()	
		 	
			


		