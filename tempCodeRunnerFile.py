import matplotlib.pyplot as plt

def plot_movements(seek_sequences, total_movements, titles, requests):
    """Plots all disk scheduling algorithm movements in one figure."""
    num_plots = len(seek_sequences)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, num_plots * 3.5))
    fig.suptitle('Disk Scheduling Algorithm Comparison', fontsize=16)

    if num_plots == 1:
        axs = [axs]  # Make it iterable if there's only one subplot

    for i, ax in enumerate(axs):
        seq = seek_sequences[i]
        title = titles[i]
        movement = total_movements[i]
        
        # Plot the head movement
        ax.plot(seq, range(len(seq)), marker='o', drawstyle='steps-mid', label=f'Head Path (Total: {movement})')
        
        # Plot the original requests as vertical lines for reference
        for req in requests:
            ax.axvline(x=req, color='grey', linestyle='--', alpha=0.5)
            
        # Plot start and end points
        ax.plot(seq[0], 0, 'go', markersize=10, label='Start') # Start
        ax.plot(seq[-1], len(seq)-1, 'rs', markersize=10, label='End') # End
        
        # Invert y-axis to show sequence from top to bottom (time flows down)
        ax.invert_yaxis()
        
        ax.set_title(title)
        ax.set_xlabel('Cylinder Number')
        ax.set_ylabel('Request Sequence (Time)')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

def calculate_movement(sequence):
    """Calculates total head movement from a seek sequence."""
    movement = 0
    for i in range(len(sequence) - 1):
        movement += abs(sequence[i+1] - sequence[i])
    return movement

def scan(requests, head, direction, disk_max=199):
    seek_sequence = [head]
    requests.sort()
    
    # Split requests into 'down' and 'up' relative to the head
    down = [r for r in requests if r < head]
    up = [r for r in requests if r >= head]
    
    if direction == 'up':
        seek_sequence.extend(up)
        seek_sequence.append(disk_max) # Go to the end
        seek_sequence.extend(down[::-1]) # Service in reverse
    else: # direction == 'down'
        seek_sequence.extend(down[::-1])
        seek_sequence.append(0) # Go to the beginning
        seek_sequence.extend(up)
        
    movement = calculate_movement(seek_sequence)
    return movement, seek_sequence

def cscan(requests, head, direction, disk_max=199):
    seek_sequence = [head]
    requests.sort()
    
    down = [r for r in requests if r < head]
    up = [r for r in requests if r >= head]
    
    if direction == 'up':
        seek_sequence.extend(up)
        seek_sequence.append(disk_max) # Go to the end
        seek_sequence.append(0) # Jump to the beginning
        seek_sequence.extend(down)
    else: # direction == 'down'
        seek_sequence.extend(down[::-1])
        seek_sequence.append(0) # Go to the beginning
        seek_sequence.append(disk_max) # Jump to the end
        seek_sequence.extend(up[::-1])

    movement = calculate_movement(seek_sequence)
    return movement, seek_sequence

def look(requests, head, direction):
    seek_sequence = [head]
    requests.sort()
    
    down = [r for r in requests if r < head]
    up = [r for r in requests if r >= head]
    
    if direction == 'up':
        seek_sequence.extend(up)
        seek_sequence.extend(down[::-1])
    else: # direction == 'down'
        seek_sequence.extend(down[::-1])
        seek_sequence.extend(up)
        
    movement = calculate_movement(seek_sequence)
    return movement, seek_sequence

def clook(requests, head, direction):
    seek_sequence = [head]
    requests.sort()
    
    down = [r for r in requests if r < head]
    up = [r for r in requests if r >= head]
    
    if direction == 'up':
        seek_sequence.extend(up)
        seek_sequence.extend(down) # Jump to the lowest, then move up
    else: # direction == 'down'
        seek_sequence.extend(down[::-1])
        seek_sequence.extend(up[::-1]) # Jump to the highest, then move down

    movement = calculate_movement(seek_sequence)
    return movement, seek_sequence

# --- Main Execution ---
if __name__ == "__main__":
    # Example data from the problem
    initial_requests = [95, 180, 34, 119, 11, 123, 62, 64]
    start_head = 50
    start_direction = 'up'
    disk_size = 199
    
    all_sequences = []
    all_movements = []
    all_titles = []
    
    # 1. SCAN
    # Note: We pass a copy of the requests list to avoid modifying it
    mov, seq = scan(list(initial_requests), start_head, start_direction, disk_size)
    all_sequences.append(seq)
    all_movements.append(mov)
    all_titles.append('SCAN Algorithm')

    # 2. CSCAN
    mov, seq = cscan(list(initial_requests), start_head, start_direction, disk_size)
    all_sequences.append(seq)
    all_movements.append(mov)
    all_titles.append('C-SCAN Algorithm')

    # 3. LOOK
    mov, seq = look(list(initial_requests), start_head, start_direction)
    all_sequences.append(seq)
    all_movements.append(mov)
    all_titles.append('LOOK Algorithm')

    # 4. C-LOOK
    mov, seq = clook(list(initial_requests), start_head, start_direction)
    all_sequences.append(seq)
    all_movements.append(mov)
    all_titles.append('C-LOOK Algorithm')

    # Print results to console (matches Task 1)
    print("--- Disk Scheduling Results ---")
    for i in range(len(all_titles)):
        print(f"{all_titles[i]:<16} | Total Movement: {all_movements[i]}")
        # print(f"Seek Sequence: {all_sequences[i]}\n") # Uncomment to see the full path
        
    # Plot all results
    plot_movements(all_sequences, all_movements, all_titles, initial_requests)