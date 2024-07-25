import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- App Title and Description ---
st.title("Thermostat Simulation with Q-Learning and PID Control")
st.write("This simulation compares Q-Learning and PID control for maintaining room temperature.")

# --- Input Parameters ---
initial_room_temperature = st.number_input("Initial Room Temperature (°C)", min_value=10, max_value=30, value=19)
outside_temperature = st.number_input("Outside Temperature (°C)", min_value=0, max_value=40, value=10)
thermostat_setting = st.number_input("Thermostat Setting (°C)", min_value=15, max_value=25, value=20)
heater_power = st.slider("Heater Power (°C/minute)", min_value=0.1, max_value=0.5, value=0.3)
heat_loss = st.slider("Heat Loss (°C/minute)", min_value=0.05, max_value=0.2, value=0.1)

# Q-learning Parameters
num_states = 41
num_actions = 2
q_table = np.zeros((num_states, num_actions))
learning_rate = 0.1
discount_factor = 0.95
exploration_rate = 0.1
episodes = st.number_input("Training Episodes (Q-Learning)", min_value=100, max_value=5000, value=1000)

# Simulation Parameters
simulation_minutes = st.number_input("Simulation Minutes", min_value=10, max_value=120, value=60)

# PID Parameters
Kp = st.slider("Kp (Proportional Gain)", min_value=0.1, max_value=2.0, value=0.5)
Ki = st.slider("Ki (Integral Gain)", min_value=0.01, max_value=0.5, value=0.1)
Kd = st.slider("Kd (Derivative Gain)", min_value=0.001, max_value=0.2, value=0.01)


# --- Helper Functions (Q-Learning) ---
def get_state(temperature):
    return int((temperature - 10) // 0.5)

def get_action(state):
    if np.random.uniform(0, 1) < exploration_rate:
        return np.random.choice(num_actions)  # Exploration
    else:
        return np.argmax(q_table[state, :])   # Exploitation
        
def get_reward(state, action, thermostat_setting):
    state_temp = 10 + state * 0.5
    if abs(state_temp - thermostat_setting) <= 0.5:
        return 10  # Within acceptable range
    elif action == 1 and state_temp > thermostat_setting + 0.5:  # Too hot
        return -10
    elif action == 0 and state_temp < thermostat_setting - 0.5:  # Too cold
        return -5
    else:
        return -1  # Slight penalty for not being in range

def run_q_learning_simulation(initial_room_temperature):
    global q_table  # Ensure we're using the global q_table
    time = []
    room_temperatures = []
    outside_temperatures = []
    heater_output = []

    for episode in range(episodes):
        room_temperature = initial_room_temperature
        state = get_state(room_temperature)
        for minute in np.arange(0, simulation_minutes, 0.1):
            action = get_action(state)
            heater_output.append(action)

            if action == 1:
                room_temperature += heater_power * 0.1
            else:
                room_temperature -= heat_loss * 0.1

            next_state = get_state(room_temperature)
            reward = get_reward(next_state, action, thermostat_setting)

            q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state

            time.append(minute)
            room_temperatures.append(room_temperature)
            outside_temperatures.append(outside_temperature)

    df = pd.DataFrame({
        'Time (Minutes)': time,
        'Room Temperature (°C)': room_temperatures,
        'Outside Temperature (°C)': outside_temperatures,
        'Heater Output': heater_output
    })

    loss_area_data = df[df['Room Temperature (°C)'] > thermostat_setting]
    return time, room_temperatures, outside_temperatures, heater_output, df, loss_area_data

# --- Simulation Logic (PID) ---
def run_pid_simulation(initial_room_temperature):
    time = []
    room_temperatures = []
    outside_temperatures = []
    heater_output = []

    integral_error = 0
    previous_error = 0
    room_temperature = initial_room_temperature

    for minute in np.arange(0, simulation_minutes, 0.1):
        time.append(minute)

        error = thermostat_setting - room_temperature
        proportional_term = Kp * error
        integral_error += error * 0.1
        integral_term = Ki * integral_error
        derivative_term = Kd * (error - previous_error) / 0.1
        previous_error = error

        pid_output = proportional_term + integral_term + derivative_term
        heater_output_percent = max(0, min(1, pid_output))
        heater_output.append(heater_output_percent)

        room_temperature += (heater_power * heater_output_percent - heat_loss) * 0.1
        outside_temperatures.append(outside_temperature)
        room_temperatures.append(room_temperature)

    df = pd.DataFrame({
        'Time (Minutes)': time,
        'Room Temperature (°C)': room_temperatures,
        'Outside Temperature (°C)': outside_temperatures,
        'Heater Output (%)': heater_output
    })

    loss_area_data = df[df['Room Temperature (°C)'] > thermostat_setting]
    return time, room_temperatures, outside_temperatures, heater_output, df, loss_area_data

# --- Main App ---
simulation_type = st.selectbox("Choose Simulation Type:", ("Q-Learning", "PID", "Both"))

if st.button("Run Simulation"):
    time_q = room_temperatures_q = outside_temperatures_q = heater_output_q = df_q = loss_area_data_q = None
    time_pid = room_temperatures_pid = outside_temperatures_pid = heater_output_pid = df_pid = loss_area_data_pid = None
    
    if simulation_type == "Q-Learning" or simulation_type == "Both":
        time_q, room_temperatures_q, outside_temperatures_q, heater_output_q, df_q, loss_area_data_q = run_q_learning_simulation(initial_room_temperature)
    
    if simulation_type == "PID" or simulation_type == "Both":
        time_pid, room_temperatures_pid, outside_temperatures_pid, heater_output_pid, df_pid, loss_area_data_pid = run_pid_simulation(initial_room_temperature)

    # --- Plotting Results ---
    plt.figure(figsize=(10, 6))

    if simulation_type == "Q-Learning" and time_q is not None:
        plt.plot(time_q, room_temperatures_q, label="Room Temperature (Q-Learning)", color="blue")
        plt.plot(time_q, heater_output_q, label="Heater Output (Q-Learning)", color="lightblue", linestyle=":")
    elif simulation_type == "PID" and time_pid is not None:
        plt.plot(time_pid, room_temperatures_pid, label="Room Temperature (PID)", color="orange")
        plt.plot(time_pid, heater_output_pid, label="Heater Output (PID)", color="coral", linestyle=":")
    elif simulation_type == "Both" and time_q is not None and time_pid is not None:
        plt.plot(time_q, room_temperatures_q, label="Room Temperature (Q-Learning)", color="blue")
        plt.plot(time_pid, room_temperatures_pid, label="Room Temperature (PID)", color="orange")
        plt.plot(time_q, heater_output_q, label="Heater Output (Q-Learning)", color="lightblue", linestyle=":")
        plt.plot(time_pid, heater_output_pid, label="Heater Output (PID)", color="coral", linestyle=":")

    if time_q is not None:
        plt.plot([time_q[0], time_q[-1]], [outside_temperature, outside_temperature], label="Outside Temperature", color="red")
    elif time_pid is not None:
        plt.plot([time_pid[0], time_pid[-1]], [outside_temperature, outside_temperature], label="Outside Temperature", color="gray")

    plt.axhline(y=thermostat_setting, color='r', linestyle='--', label="Thermostat Setting")
    plt.axhline(y=thermostat_setting + 0.5, color='g', linestyle='--', label="Upper Threshold")
    plt.axhline(y=thermostat_setting - 0.5, color='g', linestyle='--', label="Lower Threshold")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Temperature (°C) / Heater Output (%)")
    plt.title("Thermostat Simulation: Q-Learning vs. PID")
    plt.legend()
    plt.grid(axis='y')

    if time_q is not None and (simulation_type == "Q-Learning" or simulation_type == "Both"):
        plt.fill_between(time_q, room_temperatures_q, thermostat_setting, 
                         where=(np.array(room_temperatures_q) > thermostat_setting),
                         color='red', alpha=0.2, interpolate=True)
        plt.text(50, 20.2, 'Loss (Q-Learning)', color='red') 

    if time_pid is not None and (simulation_type == "PID" or simulation_type == "Both"):
        plt.fill_between(time_pid, room_temperatures_pid, thermostat_setting, 
                         where=(np.array(room_temperatures_pid) > thermostat_setting),
                         color='purple', alpha=0.2, interpolate=True)
        plt.text(50, 19.8, 'Loss (PID)', color='purple') 

    plt.yticks(np.arange(9, 22, 0.5))
    plt.ylim(9, 21)

    st.pyplot(plt)  # Display the Matplotlib plot in Streamlit

    # --- Optional: Altair Charts ---
    if st.checkbox("Show Altair Charts"):
        # Altair Chart (Q-Learning)
        if simulation_type == "Q-Learning" or simulation_type == "Both":
            # ... (Your Altair chart code for Q-Learning)
            st.altair_chart(chart_q, use_container_width=True)

        # Altair Chart (PID)
        if simulation_type == "PID" or simulation_type == "Both":
            # ... (Your Altair chart code for PID)
            st.altair_chart(chart_pid, use_container_width=True)

    # Option to show raw data
    if st.checkbox("Show Raw Data"):
        st.subheader("Simulation Data")
        if simulation_type == "Q-Learning" and df_q is not None:
            st.write(df_q)
        elif simulation_type == "PID" and df_pid is not None:
            st.write(df_pid)
        elif simulation_type == "Both":
            if df_q is not None:
                st.write("Q-Learning Data:")
                st.write(df_q)
            if df_pid is not None:
                st.write("PID Data:")
                st.write(df_pid)

