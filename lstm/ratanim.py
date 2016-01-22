import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

parser = argparse.ArgumentParser()
parser.add_argument("location_file", default="London_data_2x1000Center_bin100_pos.dat")
parser.add_argument("prediction_file", default="test.npy")
parser.add_argument("--tail", type=int, default=10)
parser.add_argument("--interval", type=int, default=50)
parser.add_argument("--save_mp4")
args = parser.parse_args()

y = np.loadtxt(args.location_file) / 3.5
pred_y = np.load(args.prediction_file).reshape((-1, 2))

def update_lines(num):
    start = max(num - args.tail, 0)
    line1.set_data(y[start:num, 0], y[start:num, 1])
    line2.set_data(pred_y[start:num, 0], pred_y[start:num, 1])
    ttl.set_text("Rat position %d/%d" % (num, y.shape[0]))
    #plt.draw()
    return line1, line2

fig = plt.figure()
line1,line2 = plt.plot([], [], [], [])
plt.xlim((np.min(y[:,0]), np.max(y[:,0])))
plt.ylim((np.min(y[:,1]), np.max(y[:,1])))
ttl = plt.title("test")

ax = plt.gca()
# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
# Put a legend below current axis
ax.legend(('Actual location', 'Predicted location'), loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

line_ani = animation.FuncAnimation(fig, update_lines, y.shape[0], interval=args.interval, blit=False)
if args.save_mp4:
  line_ani.save(args.save_mp4)
plt.show()
