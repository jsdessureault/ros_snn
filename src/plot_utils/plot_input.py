#!/usr/bin/env python 
import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from std_msgs.msg import Float32, String, Float32MultiArray, Int16

node = "node_input_plot"
rospy.init_node(node, anonymous=True)
rospy.loginfo(node)

nb_input_neurons = rospy.get_param("~input_neurons")
SNNname = rospy.get_param("~SNNname")

simulation_length = 1000
x_lim = simulation_length

volts = []
times = []
#topics_input = []
plot_array = []

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

title = "SENSORY voltage for "+SNNname 
plt.title(title)
plt.xlabel("Frames")
plt.ylim(-1.1, 1.1)
plt.ylabel("mVolts")
plt.xlim(0,x_lim)
plt.grid(True)

def init_volts_and_times():
	for neuron_nb in range (0,nb_input_neurons):
		volts.append([])
		times.append([])
	
def callback_input(data, neuron_nb):
        global simulation_length
	value = float(data.data)
	#print "CALLBACK INPUT: " + str(value)
	if len(volts[neuron_nb]) >= simulation_length:
		del times[neuron_nb][:]
		del volts[neuron_nb][:]

	# Assign voltage
	volts[neuron_nb].append(value)
	#print volts[neuron_nb]
	# Assing time and divide by simulation lenght
	times[neuron_nb].append(len(volts[neuron_nb]))
	#print times[neuron_nb]
	#print "-----"


# Initialize arrays
init_volts_and_times()

for neuron_nb in range (0,nb_input_neurons):
	rospy.loginfo('Subscribe to /'+SNNname+'_'+str(neuron_nb+1)+'_snn_in')
	#topics_input.append(rospy.Subscriber('/'+SNNname+'_'+str(neuron_nb+1)+'_snn_in', String, callbackVolts, neuron_nb))
        rospy.Subscriber("/"+SNNname+"_"+str(neuron_nb+1)+"_snn_in", String, callback_input, neuron_nb)

def update_line(num, data, ax):
	global volts, times
	#print volts[neuron_nb]
	# Get the smaller lenght of the arrays, and use only those data to plot.  If not: xdata and ydata must be the same (error). 
	smaller = 999999
	for neuron_nb in range (0, nb_input_neurons): 	
		if len(volts[neuron_nb]) < smaller:
			smaller = len(volts[neuron_nb])
  
	for neuron_nb in range (0, nb_input_neurons): 	
		#rospy.loginfo("Upadate frame: " + str(len(times[neuron_nb])) + " " + str(len(volts[neuron_nb])))	
		plot_array[neuron_nb].set_data(times[neuron_nb][:smaller], volts[neuron_nb][:smaller])
		plot_array[neuron_nb].set_label("Input neuron: " + str(neuron_nb+1))
		legend = plt.legend()

	return ax , 


# Continue defining graphic
for neuron_nb in range(0, nb_input_neurons):
	plot_tmp, = ax1.plot(times[neuron_nb], volts[neuron_nb])
	plot_array.append(plot_tmp)

# Defining animation
# Animation parameters
# - figure to plot
# - update function
# - number of frames to cache
# - interval. Delay between frames in ms. (def: 200)
# - repeat: loop animation
# - blit: quality of the animation
simulation = animation.FuncAnimation(fig1, update_line, 25, fargs=(volts,ax1), interval=50, repeat=True)
#simulation.save(filename='animation.mp4', fps=30, dpi=300)
plt.show()

rospy.spin()

