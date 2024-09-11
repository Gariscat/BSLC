import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotate_vector_around_axis(vector, axis, angle):
    """
    Rotate a given 3D vector around a specified axis by a given angle.
    
    Parameters:
    - vector: np.array of shape (3,), the vector to rotate (target position).
    - axis: np.array of shape (3,), the axis around which to rotate.
    - angle: float, the rotation angle in radians.
    
    Returns:
    - rotated_vector: np.array of shape (3,), the rotated vector.
    """
    axis = axis / np.linalg.norm(axis)  # Normalize the axis of rotation
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1.0 - cos_angle
    
    ux, uy, uz = axis
    rotation_matrix = np.array([
        [cos_angle + ux**2 * one_minus_cos, ux*uy*one_minus_cos - uz*sin_angle, ux*uz*one_minus_cos + uy*sin_angle],
        [uy*ux*one_minus_cos + uz*sin_angle, cos_angle + uy**2 * one_minus_cos, uy*uz*one_minus_cos - ux*sin_angle],
        [uz*ux*one_minus_cos - uy*sin_angle, uz*uy*one_minus_cos + ux*sin_angle, cos_angle + uz**2 * one_minus_cos]
    ])
    
    rotated_vector = np.dot(rotation_matrix, vector)
    return rotated_vector

def spin_light(position, target, axis, speed, dt):
    """
    Spin the light around a given axis at a certain speed.
    
    Parameters:
    - position: np.array of shape (3,), the fixed position of the LED light.
    - target: np.array of shape (3,), the current target position the LED points towards.
    - axis: np.array of shape (3,), the axis around which the LED light spins.
    - speed: float, the angular speed in radians per second.
    - dt: float, the time step for each update in seconds.
    
    Returns:
    - new_target: np.array of shape (3,), the new target position after the rotation.
    """
    angle = speed * dt  # Calculate the rotation angle
    direction = target - position  # Compute the direction vector from position to target
    new_direction = rotate_vector_around_axis(direction, axis, angle)  # Rotate the direction vector
    new_target = position + new_direction  # Update the target position
    return new_target

def random_spin_light(position, target, speed, dt):
    """
    Spin the light with a random axis at a constant speed.
    
    Parameters:
    - position: np.array of shape (3,), the fixed position of the LED light.
    - target: np.array of shape (3,), the current target position the LED points towards.
    - speed: float, the angular speed in radians per second.
    - dt: float, the time step for each update in seconds.
    
    Returns:
    - new_target: np.array of shape (3,), the new target position after the random rotation.
    """
    angle = speed * dt  # Calculate the rotation angle
    direction = target - position  # Compute the direction vector from position to target
    
    # Generate a random axis of rotation (random 3D unit vector)
    random_axis = np.random.randn(3)
    random_axis /= np.linalg.norm(random_axis)  # Normalize to make it a unit vector
    
    # Rotate the direction vector around the random axis
    new_direction = rotate_vector_around_axis(direction, random_axis, angle)
    
    # Update the target position
    new_target = position + new_direction
    return new_target, random_axis


if __name__ == '__main__':
    # Example: Smaller angle between the vector and axis
    position = np.array([0.0, 0.0, 0.0])   # Fixed position of the LED light
    target = np.array([1.0, 1.0, 0.5])     # Initial target position
    axis = np.array([0.0, 1.0, 0.5])       # Axis around which to spin
    axis = axis / np.linalg.norm(axis)      # Normalize the axis
    speed = np.radians(240)  # Speed in radians per second (30 degrees per second)
    dt = 0.1  # Time step for each update (100 ms)

    # Visualize the spin for a few steps
    num_steps = 20
    positions = [target]

    for i in range(num_steps):
        target = spin_light(position, target, axis, speed, dt)
        positions.append(target)

    positions = np.array(positions)

    # Create figure and 3D axis for visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the spinning target positions
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', label='Light path')

    # Plot the fixed position of the light
    ax.scatter(position[0], position[1], position[2], color='red', label='Fixed position', s=100)

    # Plot the initial vector from position to target
    ax.quiver(position[0], position[1], position[2],
            positions[0, 0], positions[0, 1], positions[0, 2],
            color='blue', label='Initial vector', arrow_length_ratio=0.1)

    # Plot the axis of rotation
    axis_length = 2  # Extend the axis for better visibility
    ax.quiver(position[0], position[1], position[2],
            axis[0] * axis_length, axis[1] * axis_length, axis[2] * axis_length,
            color='green', label='Axis of rotation', arrow_length_ratio=0.1)

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('LED Light Spinning Around an Axis with Smaller Angle')

    # Show legend
    plt.legend()
    plt.show()
    plt.close()
    
    
    num_steps = 20
    positions = [target]
    axes = []  # To store the random axes

    for i in range(num_steps):
        target, random_axis = random_spin_light(position, target, speed, dt)
        positions.append(target)
        axes.append(random_axis)

    positions = np.array(positions)
    axes = np.array(axes)

    # Create figure and 3D axis for visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the spinning target positions
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', label='Light path')

    # Plot the fixed position of the light
    ax.scatter(position[0], position[1], position[2], color='red', label='Fixed position', s=100)

    # Plot the random axes for each frame
    for i in range(len(axes)):
        ax.quiver(position[0], position[1], position[2],
                axes[i, 0], axes[i, 1], axes[i, 2],
                color='green', alpha=0.5)

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('LED Light Random Spinning with Random Axes')

    # Show legend
    plt.legend()
    plt.show()
    plt.close()  # Close the plot after showing