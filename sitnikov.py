import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.colors as colors
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D


def sitnikov_equation(y, t, e, d):
    z, v = y
    r = np.sqrt(1 + e * np.cos(t + d))
    dz_dt = v
    dv_dt = -z / (z ** 2 + r ** 2) ** (3 / 2)
    return [dz_dt, dv_dt]


def solve_sitnikov_equation(e, d, v0, num_points, num_periods):
    # Time array
    t = np.linspace(0, 2 * np.pi * num_periods, num_points)

    # Initial conditions
    z0 = 0
    # Solve the sitnikov equation
    y0 = [z0, v0]
    sol = odeint(sitnikov_equation, y0, t, args=(e, d,))

    # Extract z values
    z_values = sol[:, 0]
    v_values = sol[:, 1]

    # Find the indices where z_values changes sign
    crossing_indices = np.where(np.diff(np.sign(z_values)))[0]

    # Get the r value and dz_dt for each crossing
    r_values = np.sqrt(1 + e * np.cos(t[crossing_indices] + d))
    phases = (t[crossing_indices] + d) % (2 * np.pi)
    dz_dt_values = v_values[crossing_indices]
    time_values = t[crossing_indices]

    return z_values, r_values, phases, dz_dt_values, time_values


# Parameters
e_values = np.linspace(0.1, 0.1, 1)  # Range of e values
v0_values = np.linspace(1.3, 1.301, 120)  # Range of v0 values
num_points = 1000000  # Number of points in phase and velocity space
num_periods = 500  # Number of periods to simulate

# Create a figure
plt.figure()

# Create a colormap
cmap = cm.get_cmap('coolwarm', len(v0_values))

# Create a list to store Line2D objects for the legend
legend_lines = []
t = np.linspace(0, 2 * np.pi * num_periods, num_points)

# Loop over all combinations of e and v0
plot_phases = []
plot_dz_dt_values = []
plot_v0 = []

# Linear plot for intersections
times = []

for i, e in enumerate(e_values):
    d_values = np.linspace(1/4*np.pi, 1/4*np.pi, 1)
    for j, v0 in enumerate(v0_values):
        # if j == 0 or j == len(v0_values) - 1:
        # legend_lines.append(mlines.Line2D( # Gives small label for v_0
        #     [], [], color=cmap(j), label=f'v_0={v0:.4f}'))
        for k, d in enumerate(d_values):
            # Solve the equation and plot the solution
            z_values, r_values, phases, dz_dt_values, time_values = solve_sitnikov_equation(
                e, d, v0, num_points, num_periods)

            # Remove the first crossing
            phases = phases[1:]
            dz_dt_values = np.abs(dz_dt_values[1:])
            r_values = r_values[1:]
            time_values = time_values[1:]
            plot_phases.append(phases[0])
            times.append(time_values[0])
            plot_dz_dt_values.append(dz_dt_values[0])
            plot_v0.append(v0)

            # for r, dz_dt in zip(r_values, dz_dt_values): # Prints r and speed values
            #     print(f'r={r:.4f}, dz_dt={dz_dt:.4f}')
            #     break

            # # Use color from colormap
            # plt.plot(t, z_values, color=cmap(j))

# # Plotting the time and dz/dt plot

# times -= times[0]
# plt.scatter(times, plot_dz_dt_values, c=plot_v0, cmap=cmap)
# cbar = plt.colorbar()

# # Disable scientific notation
# cbar.formatter.set_useOffset(False)
# cbar.update_ticks()

# # Set smaller font size
# cbar.ax.tick_params(labelsize=8)

# # Set x-axis labels in ticks of 2pi
# ax = plt.gca()
# ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2*np.pi))
# ax.xaxis.set_major_formatter(ticker.FuncFormatter(
#     lambda val, pos: '{:.0f}$\pi$'.format(val/np.pi) if val != 0 else '0'))

# # Convert phases, times, and dz_dt_values to 2D arrays
# phases = np.array(plot_phases)
# dz_dt_values = np.array(plot_dz_dt_values)
# times = np.array(times)

# x = dz_dt_values * np.cos(phases)
# y = dz_dt_values * np.sin(phases)

# # # Create polar plot # # #

# Add labels and legend
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')

# Plot the scatter plot
scatter = ax.scatter(plot_phases, plot_dz_dt_values, c=plot_v0, cmap=cmap)
# ax.set_rmin(1.37)
# ax.set_rmax(1.44)
ax.set_xlabel('Phase')
# ax.set_zlabel('Time (normalized)')
ax.set_title(
    'A polar plot of the phase space associated with\nthe Sitnikov problem on first crossing, zoomed in.', fontsize=12)
cbar = fig.colorbar(scatter)
cbar.set_label('$v_0$', fontsize=11)
cbar.ax.tick_params(labelsize=8)
cbar.formatter.set_useOffset(False)  # Disable scientific notation
cbar.update_ticks()

# Set colorbar tick format to display at most 2 decimal places
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

# plt.xlabel('Time (normalized)')
# plt.ylabel('z')
# plt.ylabel('|dz/dt| on first cross')
# plt.title('A scatter plot of the speed and normalized time\nof first crossing. The colors represent initial speed $v_0$.', fontsize=11)
plt.legend(handles=legend_lines, loc='upper right', fontsize='x-small')

# Show the plot
plt.show()
