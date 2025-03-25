import matplotlib.pyplot as plt
import numpy as np

# Define the tasks with fixed start positions - extended timeline (200 days)
tasks = {
    "Data Collection": [
        {"name": "HAM10K", "duration": 10, "start": 5},
        {"name": "Preprocessing", "duration": 18, "start": 15}
    ],
    "CNN Model Training": [
        {"name": "Architecture", "duration": 15, "start": 35},
        {"name": "Training", "duration": 20, "start": 50},
        {"name": "Tuning", "duration": 15, "start": 70}
    ],
    "Retrieval Module": [
        {"name": "Vector DB", "duration": 18, "start": 85},
        {"name": "Graph DB", "duration": 15, "start": 103},
        {"name": "Integration", "duration": 12, "start": 118}
    ],
    "LLM Integration": [
        {"name": "Model", "duration": 10, "start": 135},
        {"name": "Pipeline", "duration": 12, "start": 145},
        {"name": "Optimization", "duration": 13, "start": 157}
    ],
    "Testing & Validation": [
        {"name": "Unit Tests", "duration": 15, "start": 175},
        {"name": "Integration", "duration": 15, "start": 190}
    ]
}

# Distinct colors for each subtask type
colors = {
    "HAM10K": "#3498db",       # blue
    "Preprocessing": "#e74c3c",   # red
    "Architecture": "#2ecc71",    # green
    "Training": "#9b59b6",        # purple
    "Tuning": "#f39c12",          # orange
    "Vector DB": "#1abc9c",       # teal
    "Graph DB": "#d35400",        # dark orange
    "Integration": "#7f8c8d",     # gray
    "Model": "#f1c40f",     # yellow
    "Pipeline": "#34495e",        # navy
    "Optimization": "#16a085",    # green blue
    "Unit Tests": "#e67e22",      # orange
    "Integration": "#8e44ad"      # violet
}

fig, ax = plt.subplots(figsize=(15, 5))
y_labels = []

# Plot each task with subtasks on the same row
for i, (task, subtasks) in enumerate(tasks.items()):
    y_labels.append(task)
    
    for subtask in subtasks:
        bar = ax.barh(
            i, 
            subtask["duration"], 
            left=subtask["start"], 
            color=colors[subtask["name"]],
            edgecolor='black', 
            alpha=0.8
        )
        
        # Add text labels within the bars if enough space
        if subtask["duration"] >= 10:
            ax.text(
                subtask["start"] + subtask["duration"]/2, 
                i,
                f"{subtask['name']}",
                va='center', 
                ha='center', 
                fontsize=8,
                fontweight='bold',
                color='black'
            )

# Customize plot
ax.set_yticks(range(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=10, fontweight='bold')
ax.set_xlabel("Days", fontsize=12)
ax.set_title("Project Timeline - Gantt Chart", fontsize=14, fontweight='bold')
ax.invert_yaxis()  # Invert y-axis for better readability
ax.grid(axis='x', linestyle='--', alpha=0.6)

# Set x-axis limits with padding
ax.set_xlim(0, 220)

# Add vertical lines for month markers (assuming ~30 days per month)
for x in range(0, 211, 30):
    ax.axvline(x=x, color='gray', linestyle='-', alpha=0.3)
    ax.text(x, -0.5, f"M{x//30}", ha='center', fontsize=8)

# Create legend with reduced size and position at top right
legend_handles = [plt.Rectangle((0,0), 1, 1, color=color, ec="black") 
                 for name, color in colors.items()]
legend = ax.legend(legend_handles, colors.keys(), 
          loc='upper right',
          ncol=2, fontsize=7)

# Set x-ticks to show every 20 days
ax.set_xticks(np.arange(0, 211, 20))

plt.tight_layout()
plt.show()
