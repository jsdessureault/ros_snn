#!/usr/bin/env python 
import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from std_msgs.msg import Float32

node = "node_spikes_plot"
rospy.init_node(node, anonymous=True)
rospy.loginfo(node)

nb_output_neurons = rospy.get_param("~motor_neurons")
SNNname = rospy.get_param("~SNNname")

x_lim = 50
y_lim = 10
nb_spikes = []
times = []
iTime = []
topics_motor_spikes = []
plot_array = []

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

plt.title("Spikes for "+SNNname)
plt.ylim(0, y_lim)
plt.ylabel("Spikes")
plt.xlabel("Simulations through time")
plt.xlim(0,x_lim)
plt.grid(True)
#legend = plt.legend()

def init_iTime():
	for neuron_nb in range (0,nb_output_neurons):
		iTime.append(0)

def init_spikes_and_times():
	for neuron_nb in range (0,nb_output_neurons):
		nb_spikes.append([])
		times.append([])
	
def callbackNbSpike(data, neuron_nb):
	if iTime[neuron_nb] >= x_lim:	
		del times[neuron_nb][:]
		del nb_spikes[neuron_nb][:]
		iTime[neuron_nb] = 0
	nb_spikes[neuron_nb].append(data.data)
	times[neuron_nb].append(iTime[neuron_nb])
	iTime[neuron_nb] += 1


# Initialize arrays
init_iTime()
init_spikes_and_times()

for neuron_nb in range (0,nb_output_neurons):
	topics_motor_spikes.append(rospy.Subscriber('/motor_spikes_'+SNNname+str(neuron_nb+1), Float32, callbackNbSpike, neuron_nb))

def update_line(num, data, ax):
	# Get the smaller lenght of the arrays, and use only those data to plot.  If not: xdata and ydata must be the same (error). 
	smaller = 999
	for neuron_nb in range (0, nb_output_neurons): 	
		if len(nb_spikes[neuron_nb]) < smaller:
			smaller = len(nb_spikes[neuron_nb])
  
	for neuron_nb in range (0, nb_output_neurons): 	
		#rospy.loginfo("Upadate frame: " + str(len(times[neuron_nb])) + " " + str(len(volts[neuron_nb])))		
		plot_array[neuron_nb].set_data(times[neuron_nb][:smaller], nb_spikes[neuron_nb][:smaller])
		plot_array[neuron_nb].set_label("Neuron: " + str(neuron_nb+1))
		legend = plt.legend()
	return ax,


# Continue defining graphic
for neuron_nb in range(0, nb_output_neurons):
	plot_tmp, = ax1.plot(times[neuron_nb], nb_spikes[neuron_nb])	
	plot_array.append(plot_tmp)

# Defining animation
# Animation parameters
# - figure to plot
# - update function
# - number of frames to cache
# - interval. Delay between frames in ms. (def: 200)
# - repeat: loop animation
# - blit: quality of the animation
simulation = animation.FuncAnimation(fig1, update_line, 25, fargs=(nb_spikes,ax1), interval=50, repeat=True)
#simulation.save(filename='animation.mp4', fps=30, dpi=300)
plt.show()

#rospy.spin()

