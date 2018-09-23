import numpy as np
import pulp
import argparse
import matplotlib.pyplot as plt

class MDP:
	def __init__(self,MDP_file=""):
		self.MDP_file=MDP_file
		self.text_file = open(self.MDP_file, "r")
		self.lines = self.text_file.read().split('\n')
		self.states=int(self.lines[0])
		self.actions=int(self.lines[1])
		
		self.policy=np.random.randint(0,self.actions-1,size=self.states)
		
		self.value_function=np.zeros(self.states)
		self.action_value_function=np.zeros([self.actions,self.states],np.float64)


		self.reward_array = self.get_Matrix(begining=2)
		
		self.transition_array = self.get_Matrix(begining=2+self.states*self.actions)

		self.gamma=float(self.lines[2+2*(self.actions*self.states)])
		if self.lines[3+2*(self.actions*self.states)] == 'episodic':
			self.continuous=False
		else:
			self.continuous=True	
		self.text_file.close()
			
	def get_Matrix(self,begining=0):
		data=[]
		for i in range(self.states*self.actions):
			data.append(map(float,str.split(self.lines[i+begining],'\t')[:-1]))
		return np.resize(np.array(data),[self.states,self.actions,self.states])
	
		
	def policy_iteration(self):
		equality=False
		while not equality:	
			self.update_V()
			new_policy=np.argmax(self.action_value_function,axis=0)
			equality=np.array_equal(new_policy,self.policy)
			self.policy=new_policy
		self.get_Q()	
		
	def update_V(self):
		self.difference=3
		while self.difference >=0.00000000000001:
			for s in range(self.states):
				for a in range(self.actions):
					self.action_value_function[a,s]=np.sum(np.multiply(self.transition_array[s,a,:],self.reward_array[s,a,:]+self.gamma*self.value_function))
			new_array=np.zeros(self.states)
			for s in range(self.states):
				new_array[s]=self.action_value_function[self.policy[s],s]
			self.difference=	np.sum(np.absolute(new_array-self.value_function))
			self.value_function=new_array
		


	def LP(self):
		self.prob = pulp.LpProblem('MDP_using_Linear_programming', pulp.LpMinimize)

		self.descision_variables=[]
		for i in range(self.states):
			variable = str('V' + str(i))
			variable = pulp.LpVariable(str(variable))
			self.descision_variables.append(variable)
			
		
		self.objective=""
		for elem in self.descision_variables:
			self.objective+= elem	
		self.prob+=self.objective
		
		for i in range(self.states):
			if np.array_equal(self.transition_array[i,:,i],np.ones(self.actions)):
				self.prob+=(self.descision_variables[i]==0)
			else:
				if not self.continuous and i==self.states-1:
					self.prob+=(self.descision_variables[i]==0)
				else:			
					for j in range(self.actions):
						belman_update=0
						for t in range(self.states):
							belman_update+=self.transition_array[i,j,t]*(self.reward_array[i,j,t]+self.gamma*self.descision_variables[t])
							
						self.prob+= (self.descision_variables[i] >= belman_update)
					

		optimization_result = self.prob.solve()
		assert optimization_result == pulp.LpStatusOptimal
		
		for v in self.prob.variables():
			self.value_function[int(v.name.replace('V',''))]=v.varValue
		self.get_Q()

		
	def get_Q(self):
		for s in range(self.states):
			for a in range(self.actions):

				self.action_value_function[a,s]=np.sum(np.multiply(self.transition_array[s,a,:],self.reward_array[s,a,:]+self.gamma*self.value_function))
		self.policy=np.argmax(self.action_value_function,axis=0)
		
		
		

		
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--algorithm")
	parser.add_argument("--mdp")
	
	args=parser.parse_args()
	if args.algorithm == "lpi":
		obj=MDP(MDP_file=args.mdp)
		obj.LP()
		for i in range(obj.states):
			print str(obj.value_function[i]) + "\t" + str(obj.policy[i])+ "\n"
	if args.algorithm == "hpi":
		obj=MDP(MDP_file=args.mdp)
		obj.policy_iteration()
		for i in range(obj.states):
			print str(obj.value_function[i]) + "\t" + str(obj.policy[i])+ "\n"
	return

main()	
		 	
			


		