import matplotlib.pyplot as plt

"""
{'Fri 11-1pm': [('DTS2301', 'DTS')],
 'Fri 1:45-3:45pm': [('CSC2303', 'GROUP B')],
 'Fri 4-6pm': [],
 'Fri 8-10am': [('STA121', 'INS')],
 'Mon 11-1pm': [('MTH2301', 'GROUP B'), ('CSC2305', 'GROUP A')],
 'Mon 1:45-3:45pm': [('ITC2203', 'ITC', 'INS'), ('MTH2301', 'GROUP A')],
 'Mon 4-6pm': [],
 'Mon 8-10am': [],
 'Thu 11-1pm': [],
 'Thu 1:45-3:45pm': [('GEN2203', 'ALL')],
 'Thu 4-6pm': [('CYB2203', 'CBS'), ('CSC2305', 'GROUP B')],
 'Thu 8-10am': [('CSC2201', 'CSC')],
 'Tue 11-1pm': [],
 'Tue 1:45-3:45pm': [('CYB2301', 'CBS')],
 'Tue 4-6pm': [('GEN2201', 'ALL')],
 'Tue 8-10am': [],
 'Wed 11-1pm': [('CSC2303', 'GROUP A')],
 'Wed 1:45-3:45pm': [('DTS2303', 'DTS'), ('SWE2305', 'SWE')],
 'Wed 4-6pm': [],
 'Wed 8-10am': [('ITC2201', 'CSC', 'ITC', 'CBS', 'INS')]}
"""

# Data for the timetable
rows = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
columns = ["9:00 - 10:00", "10:00 - 11:00", "11:00 - 12:00", "12:00 - 1:00", "2:00 - 3:00"]

data = [
    ["Math", "Science", "English", "History", "Art"],
    ["PE", "Math", "Geography", "Computer", "Music"],
    ["Science", "English", "Math", "Art", "History"],
    ["History", "PE", "Science", "English", "Geography"],
    ["Art", "Music", "PE", "Math", "Computer"],
]

# Plotting an empty figure
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")  # Turn off the axes

# Create the table
table = plt.table(
    cellText=data,
    colLabels=columns,
    rowLabels=rows,
    loc="center",
    cellLoc="center",
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(col=list(range(len(columns))))  # Adjust column width

# Display the table
plt.show()