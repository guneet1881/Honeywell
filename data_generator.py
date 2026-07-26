"""
Honeywell QCS Hackathon Challenge: Grade Change Intelligence
Module 1: Synthetic Industrial Paper Mill Dataset Generator

This script generates a highly realistic time-series dataset simulating Honeywell Quality 
Control System (QCS) and Distributed Control System (DCS) logs during normal operations 
and automated paper grade transitions.

It incorporates:
- Multivariable dynamic interactions (Stock flow, Filler flow, Steam pressure, Machine speed)
- Scanner quality variables (Basis Weight, Moisture, Ash content, Caliper thickness)
- Hidden correlated DCS disturbances (Wire Vacuum Pressure, Shower Water Temp, Headbox Slice Ratio)
- Latency & sensor noise representative of industrial manufacturing environments
- Labeled Off-Spec risk events (>2.5% Basis Weight deviation from target setpoint)
- Historical success & failure transition scenarios for ML pattern mining
"""

import os
import numpy as np
import pandas as pd

# Set Random Seed for Reproducibility
np.random.seed(42)

def generate_paper_mill_dataset(n_days=14, sample_interval_sec=30, output_filename="paper_mill_grade_change_data.csv"):
    print(f"[*] Generating {n_days} days of industrial QCS paper mill data at {sample_interval_sec}s intervals...")
    
    total_steps = int((n_days * 24 * 3600) / sample_interval_sec)
    timestamps = pd.date_range(start="2026-07-01 00:00:00", periods=total_steps, freq=f"{sample_interval_sec}s")
    
    # Define standard paper grades and their typical target process setpoints
    grades = {
        "Grade_D12 (65 GSM - Newsprint)":  {"bw_sp": 65.0,  "speed": 1100, "stock": 420.0, "filler": 38.0, "steam": 320.0, "moisture_sp": 5.2, "ash_sp": 10.0, "caliper_sp": 82.0},
        "Grade_A32 (80 GSM - Standard)":   {"bw_sp": 80.0,  "speed": 950,  "stock": 510.0, "filler": 52.0, "steam": 380.0, "moisture_sp": 5.8, "ash_sp": 14.0, "caliper_sp": 104.0},
        "Grade_B18 (120 GSM - Heavy)":     {"bw_sp": 120.0, "speed": 780,  "stock": 740.0, "filler": 85.0, "steam": 490.0, "moisture_sp": 6.3, "ash_sp": 18.5, "caliper_sp": 148.0},
        "Grade_C45 (160 GSM - Board/Coated)":{"bw_sp": 160.0,"speed": 620,  "stock": 980.0, "filler": 125.0,"steam": 610.0, "moisture_sp": 6.8, "ash_sp": 21.0, "caliper_sp": 195.0}
    }
    
    grade_names = list(grades.keys())
    
    # Initialize data arrays
    current_grade = grade_names[1] # Start at Grade A32
    target_grade = current_grade
    is_transitioning = False
    transition_step = 0
    transition_duration = 0
    time_since_transition_min = 0.0
    batch_id = 1001
    
    # Current state trackers
    cur_bw = grades[current_grade]["bw_sp"]
    cur_bw_sp = grades[current_grade]["bw_sp"]
    cur_speed = float(grades[current_grade]["speed"])
    cur_stock = grades[current_grade]["stock"]
    cur_filler = grades[current_grade]["filler"]
    cur_steam = grades[current_grade]["steam"]
    cur_moisture = grades[current_grade]["moisture_sp"]
    cur_ash = grades[current_grade]["ash_sp"]
    cur_caliper = grades[current_grade]["caliper_sp"]
    
    # Auxiliary DCS sensors (Hidden correlation sources)
    cur_vacuum_press = 45.0 # kPa on forming wire
    cur_shower_temp = 48.0  # Celsius shower water
    cur_slice_ratio = 1.02  # Headbox jet-to-wire ratio
    
    data = []
    
    for i in range(total_steps):
        # Determine if we should trigger a grade transition (every ~8 to 12 hours)
        steps_per_hour = 3600 / sample_interval_sec
        if not is_transitioning and (i > 0) and (i % int(np.random.uniform(8, 14) * steps_per_hour) == 0):
            # Pick a new grade different from current
            possible_targets = [g for g in grade_names if g != current_grade]
            target_grade = np.random.choice(possible_targets)
            is_transitioning = True
            transition_step = 0
            # Transitions typically take between 15 to 35 minutes (30 to 70 steps at 30s)
            transition_duration = int(np.random.uniform(30, 70))
            batch_id += 1
            time_since_transition_min = 0.0

        # Update transition kinetics and coordinated ramping
        if is_transitioning:
            transition_step += 1
            time_since_transition_min = (transition_step * sample_interval_sec) / 60.0
            alpha = min(1.0, transition_step / transition_duration)
            
            # Smooth S-curve (sigmoid) ramping for physical setpoints
            s_alpha = 1 / (1 + np.exp(-10 * (alpha - 0.5)))
            
            target_specs = grades[target_grade]
            start_specs = grades[current_grade]
            
            # Update setpoint immediately or step-wise
            cur_bw_sp = target_specs["bw_sp"]
            
            # Ramping primary actuator setpoints
            cur_stock = start_specs["stock"] + (target_specs["stock"] - start_specs["stock"]) * s_alpha
            cur_filler = start_specs["filler"] + (target_specs["filler"] - start_specs["filler"]) * s_alpha
            cur_steam = start_specs["steam"] + (target_specs["steam"] - start_specs["steam"]) * s_alpha
            cur_speed = start_specs["speed"] + (target_specs["speed"] - start_specs["speed"]) * s_alpha
            
            # Simulate historical transition quality scenarios (Success vs Failure)
            # 40% of transitions experience severe oscillations or hidden disturbance resonance (Failures/Off-Spec)
            is_problematic_transition = (batch_id % 5 in [0, 2])
            
            if is_problematic_transition:
                # Introduce HIDDEN CORRELATION #1: Filler/Stock Resonance & Steam Lag
                # During rapid speed reduction or heavy GSM transition, filler retention efficiency fluctuates
                filler_ratio = cur_filler / max(1.0, cur_stock)
                steam_gradient_lag = np.sin(alpha * np.pi * 3) * 25.0
                cur_steam += steam_gradient_lag
                
                # Introduce HIDDEN CORRELATION #2: Wire Vacuum Drop & Thermal Instability
                cur_vacuum_press = 45.0 - (12.0 * np.sin(alpha * np.pi * 2)) # Unstable vacuum
                cur_shower_temp = 48.0 + (5.5 * np.cos(alpha * np.pi))
                
                # Impact on Basis Weight: Unmanaged hidden correlations push BW off-spec (> 2.5%)
                bw_disturbance = (target_specs["bw_sp"] * 0.038 * np.sin(alpha * np.pi * 2.5)) + (4.0 * (cur_vacuum_press < 38.0))
                cur_bw = start_specs["bw_sp"] + (target_specs["bw_sp"] - start_specs["bw_sp"]) * s_alpha + bw_disturbance
                
                # Related quality variable instability
                cur_moisture = target_specs["moisture_sp"] + np.random.normal(0, 0.4)
                cur_ash = target_specs["ash_sp"] + np.random.normal(0, 1.2)
                cur_caliper = target_specs["caliper_sp"] * (cur_bw / max(1.0, cur_bw_sp)) + np.random.normal(0, 3.5)
                
            else:
                # Smooth, successful transition (within operational limits)
                cur_vacuum_press = 45.0 + np.random.normal(0, 0.5)
                cur_shower_temp = 48.0 + np.random.normal(0, 0.3)
                cur_bw = start_specs["bw_sp"] + (target_specs["bw_sp"] - start_specs["bw_sp"]) * s_alpha + np.random.normal(0, 0.6)
                cur_moisture = start_specs["moisture_sp"] + (target_specs["moisture_sp"] - start_specs["moisture_sp"]) * s_alpha + np.random.normal(0, 0.1)
                cur_ash = start_specs["ash_sp"] + (target_specs["ash_sp"] - start_specs["ash_sp"]) * s_alpha + np.random.normal(0, 0.3)
                cur_caliper = start_specs["caliper_sp"] + (target_specs["caliper_sp"] - start_specs["caliper_sp"]) * s_alpha + np.random.normal(0, 1.0)

            # Check if transition ended
            if transition_step >= transition_duration:
                is_transitioning = False
                current_grade = target_grade
                time_since_transition_min = 0.0
                
        else:
            # Steady State Operation (Normal running with minor white noise and cyclical disturbances)
            target_specs = grades[current_grade]
            cur_bw_sp = target_specs["bw_sp"]
            
            # Minor random walk around steady setpoints
            cur_stock = target_specs["stock"] + np.random.normal(0, 1.5)
            cur_filler = target_specs["filler"] + np.random.normal(0, 0.5)
            cur_steam = target_specs["steam"] + np.random.normal(0, 2.0)
            cur_speed = target_specs["speed"] + np.random.normal(0, 1.0)
            
            cur_vacuum_press = 45.0 + np.random.normal(0, 0.4)
            cur_shower_temp = 48.0 + np.random.normal(0, 0.2)
            cur_slice_ratio = 1.02 + np.random.normal(0, 0.005)
            
            cur_bw = target_specs["bw_sp"] + np.random.normal(0, 0.4)
            cur_moisture = target_specs["moisture_sp"] + np.random.normal(0, 0.08)
            cur_ash = target_specs["ash_sp"] + np.random.normal(0, 0.2)
            cur_caliper = target_specs["caliper_sp"] + np.random.normal(0, 0.8)
            time_since_transition_min = 0.0

        # Simulate real-world sensor latency & occasional missing data (Edge case handling)
        # 0.5% chance of momentary sensor drop (NaN or spike) to prove architecture viability
        is_sensor_dropout = (np.random.random() < 0.003)
        logged_bw = np.nan if is_sensor_dropout else round(cur_bw, 2)

        # Calculate Basis Weight Deviation %
        bw_dev_pct = round(((cur_bw - cur_bw_sp) / cur_bw_sp) * 100.0, 3)
        
        # Binary target: Off-Spec Risk (>2.5% deviation from setpoint)
        # We classify as risk if current deviation > 2.5% or if it's a problematic transition trending towards off-spec
        is_off_spec = int(abs(bw_dev_pct) >= 2.5)

        # Derived XAI & Correlation Discovery features (to be mined by AI)
        stock_filler_ratio = round(cur_stock / max(1.0, cur_filler), 3)
        steam_to_speed_ratio = round(cur_steam / max(1.0, cur_speed), 4)

        data.append({
            "Timestamp": timestamps[i],
            "Batch_ID": batch_id,
            "Current_Grade": current_grade,
            "Target_Grade": target_grade,
            "Is_Transitioning": int(is_transitioning),
            "Time_Since_Transition_Min": round(time_since_transition_min, 2),
            "Stock_Flow_L_min": round(cur_stock, 2),
            "Filler_Flow_L_min": round(cur_filler, 2),
            "Dryer_Steam_Pressure_kPa": round(cur_steam, 2),
            "Machine_Speed_m_min": round(cur_speed, 1),
            "Wire_Vacuum_Pressure_kPa": round(cur_vacuum_press, 2), # Hidden correlation sensor 1
            "Shower_Water_Temp_C": round(cur_shower_temp, 2),       # Hidden correlation sensor 2
            "Headbox_Slice_Ratio": round(cur_slice_ratio, 4),       # Hidden correlation sensor 3
            "Stock_Filler_Ratio": stock_filler_ratio,
            "Steam_Speed_Ratio": steam_to_speed_ratio,
            "Moisture_Pct": round(cur_moisture, 2),
            "Ash_Content_Pct": round(cur_ash, 2),
            "Caliper_um": round(cur_caliper, 1),
            "Basis_Weight_Setpoint_GSM": round(cur_bw_sp, 2),
            "Basis_Weight_Actual_GSM": logged_bw,
            "Basis_Weight_Deviation_Pct": bw_dev_pct,
            "Risk_Off_Spec_2_5_Pct": is_off_spec
        })
        
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"[+] Dataset successfully generated and saved to '{output_filename}' ({len(df)} records, {len(df.columns)} variables).")
    print(f"[+] Total Grade Transition Events Simulated: {df['Batch_ID'].nunique() - 1}")
    print(f"[+] Total Off-Spec Risk Records (>2.5% deviation): {df['Risk_Off_Spec_2_5_Pct'].sum()} ({round(df['Risk_Off_Spec_2_5_Pct'].mean()*100, 2)}%)")
    
    return df

if __name__ == "__main__":
    generate_paper_mill_dataset()
