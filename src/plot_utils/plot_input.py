#!/usr/bin/env python 
import rospy
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from std_msgs.msg import Float32, String, Float32MultiArray, Int16
from lxml import etree

node = "node_input_plot"
rospy.init_node(node, anonymous=True)
rospy.loginfo(node)

SNNname = rospy.get_param("~SNNname")
pathSNN = rospy.get_param("~path")
xml = rospy.get_param("~xml")

simulation_length = 1000
x_lim = simulation_length

volts = []
times = []

#topics_input = []
plot_array = []

sensory_neurons = 0
inter_neurons = 0
motor_neurons = 0 
inter_layers = 0

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

title = "SENSORY voltage for "+SNNname 
plt.title(title)
plt.xlabel("Frames")
plt.ylim(0, 1.1)
plt.ylabel("volts")
plt.xlim(0,x_lim)
plt.grid(True)

def count_neurons():
    global tree, sensory_neurons, inter_neurons, motor_neurons, inter_layers
    # Defining number of each layer
    for neuron in tree.xpath("/SNN/layer/neuron"):
        layer_type = neuron.getparent().get("type")
        if layer_type == "sensory":
            sensory_neurons+=1
        if layer_type == "inter":
            inter_neurons+=1
            inter_layers = 1
        if layer_type == "motor":
            motor_neurons+=1

def init_volts_and_times():
	for neuron_nb in range (0,sensory_neurons):
		volts.append([])
		times.append([])
        

def reinit_if_needed(neuron_nb):
	global simulation_length
	if len(volts[neuron_nb]) >= simulation_length:
		del times[neuron_nb][:]
		del volts[neuron_nb][:]
        
    
def callback_input(data, neuron_nb):
	global simulation_length
	value = float(data.data)
	#print "CALLBACK INPUT: " + str(value)
	reinit_if_needed(neuron_nb)
	volts[neuron_nb].append(value)
	times[neuron_nb].append(len(volts[neuron_nb]))


xml_file = pathSNN + "xml/" + xml
rospy.loginfo("xml:" + xml_file)
try:
    tree = etree.parse(xml_file)
    count_neurons()
except:
    rospy.loginfo("Error XML file: " + xml_file)    
    sys.exit(1)
    
# Initialize arrays
init_volts_and_times()

for neuron_nb in range (0,sensory_neurons):
	rospy.loginfo('Subscribe to /'+SNNname+'_'+str(neuron_nb+1)+'_snn_in')
	#topics_input.append(rospy.Subscriber('/'+SNNname+'_'+str(neuron_nb+1)+'_snn_in', String, callbackVolts, neuron_nb))
        rospy.Subscriber("/"+SNNname+"_"+str(neuron_nb+1)+"_snn_in", String, callback_input, neuron_nb)

def update_line(num, data, ax):
	global volts, times
	#print volts[neuron_nb]
	# Get the smaller lenght of the arrays, and use only those data to plot.  If not: xdata and ydata must be the same (error). 
	smaller = 999999
	for neuron_nb in range (0, sensory_neurons): 	
		if len(volts[neuron_nb]) < smaller:
			smaller = len(volts[neuron_nb])
  
	for neuron_nb in range (0, sensory_neurons): 	
		#rospy.loginfo("Upadate frame: " + str(len(times[neuron_nb])) + " " + str(len(volts[neuron_nb])))	
		plot_array[neuron_nb].set_data(times[neuron_nb][:smaller], volts[neuron_nb][:smaller])
		plot_array[neuron_nb].set_label("Neuron: " + str(neuron_nb+1))
		legend = plt.legend()

	return ax , 


# Continue defining graphic
for neuron_nb in range(0, sensory_neurons):
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

