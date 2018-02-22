#!/usr/bin/env python 
import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from std_msgs.msg import Float32, Float32MultiArray, Int16
from lxml import etree

node = "node_volt_plot"
rospy.init_node(node, anonymous=True)
rospy.loginfo(node)

SNNname = rospy.get_param("~SNNname")
pathSNN = rospy.get_param("~path")
xml = rospy.get_param("~xml")

simulation_length = 1

x_lim = 1
old_x_lim = x_lim

volts = []
times = []
iTime = []
topics_motor = []
plot_array = []

sensory_neurons = 0
inter_neurons = 0
motor_neurons = 0 
inter_layers = 0

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

title = "MOTOR voltage for "+SNNname
plt.title(title)
plt.xlabel("Frames")
plt.ylim(-0.2, 1.2)
plt.ylabel("mVolts")
plt.xlim(0,x_lim)
plt.grid(True)
#legend = plt.legend()


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

def init_iTime():
	for neuron_nb in range (0,motor_neurons):
		iTime.append(0)

def init_volts_and_times():
	for neuron_nb in range (0,motor_neurons):
		volts.append([])
		times.append([])
	
def callbackVolts(data, neuron_nb):
	del times[neuron_nb][:]
	del volts[neuron_nb][:]
	global x_lim, old_x_lim, fig1, simulation_length

	# Ajust the limit.  
	x_lim = len(data.data)

	# If the limit has changed, modify ploting scale (x_limit)	
	#print "x_lim: " + str(x_lim) + " " + "old_x_lim: " + str(old_x_lim)
	if x_lim != old_x_lim:
		old_x_lim = x_lim
		plt.xlim(0,x_lim)

	# Assign voltage
	for j in range(0,len(data.data)):
		volts[neuron_nb].append(data.data[j])

	# Assing time and divide by simulation lenght
	times[neuron_nb].append(range(0,len(volts[neuron_nb])))
	#for j in range(0, len(data.data)):
	#	times[neuron_nb][0][j] /= simulation_lenght


		
def callbackSimulationLenght(data):
	global simulation_length, fig1
	simulation_length = data.data
	plt.title(title + " Simulation length: " + str(simulation_length))


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

for neuron_nb in range (0,motor_neurons):
	topics_motor.append(rospy.Subscriber('/motor_volts_'+SNNname+str(neuron_nb+1), Float32MultiArray, callbackVolts, neuron_nb))

topic_simulation_lenght = rospy.Subscriber('/topic_simulation_lenght_'+SNNname, Int16, callbackSimulationLenght)

def update_line(num, data, ax):

	# Get the smaller lenght of the arrays, and use only those data to plot.  If not: xdata and ydata must be the same (error). 
	smaller = 999999
	for neuron_nb in range (0, motor_neurons): 	
		if len(volts[neuron_nb]) < smaller:
			smaller = len(volts[neuron_nb])
  
	for neuron_nb in range (0, motor_neurons): 	
		#rospy.loginfo("Upadate frame: " + str(len(times[neuron_nb])) + " " + str(len(volts[neuron_nb])))		
		plot_array[neuron_nb].set_data(times[neuron_nb][:smaller], volts[neuron_nb][:smaller])
		plot_array[neuron_nb].set_label("Neuron: " + str(neuron_nb+1))
		legend = plt.legend()
	return ax,



# Continue defining graphic
for neuron_nb in range(0, motor_neurons):
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

#rospy.spin()

