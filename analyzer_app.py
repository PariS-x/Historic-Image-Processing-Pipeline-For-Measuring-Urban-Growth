import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# --- Main Application Class ---
class ChangeAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Satellite Image Change Analyzer")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1f232b") # Dark background

        # --- File paths ---
        self.before_path = ""
        self.after_path = ""

        # --- Main layout frames ---
        self.control_frame = tk.Frame(root, width=300, bg="#4a5568", bd=2, relief="solid")
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)
        self.control_frame.pack_propagate(False) # Prevent frame from shrinking

        self.plot_frame = tk.Frame(root, bg="#2d3748")
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # --- Title ---
        tk.Label(self.control_frame, text="Controls", font=("Helvetica", 20, "bold"), bg="#4a5568", fg="white").pack(pady=10)

        # --- File Upload ---
        self.before_btn = ttk.Button(self.control_frame, text="Upload 'Before' Image", command=self.load_before)
        self.before_btn.pack(pady=10, padx=20, fill="x")
        self.before_label = tk.Label(self.control_frame, text="No file selected", font=("Helvetica", 8), bg="#4a5568", fg="gray")
        self.before_label.pack(padx=20)

        self.after_btn = ttk.Button(self.control_frame, text="Upload 'After' Image", command=self.load_after)
        self.after_btn.pack(pady=10, padx=20, fill="x")
        self.after_label = tk.Label(self.control_frame, text="No file selected", font=("Helvetica", 8), bg="#4a5568", fg="gray")
        self.after_label.pack(padx=20)

        # --- Tuning Parameters ---
        tk.Label(self.control_frame, text="Change Threshold", font=("Helvetica", 12, "bold"), bg="#4a5568", fg="white").pack(pady=(20, 5))
        self.threshold_slider = tk.Scale(self.control_frame, from_=1, to=254, orient="horizontal", length=260, bg="#4a5568", fg="white", troughcolor="#2d3748", highlightthickness=0)
        self.threshold_slider.set(70)
        self.threshold_slider.pack(padx=20)
        tk.Label(self.control_frame, text="Higher = Detects only major changes", font=("Helvetica", 8), bg="#4a5568", fg="gray").pack()


        tk.Label(self.control_frame, text="Noise Cleaning (Kernel)", font=("Helvetica", 12, "bold"), bg="#4a5568", fg="white").pack(pady=(20, 5))
        self.kernel_slider = tk.Scale(self.control_frame, from_=3, to=21, resolution=2, orient="horizontal", length=260, bg="#4a5568", fg="white", troughcolor="#2d3748", highlightthickness=0)
        self.kernel_slider.set(9)
        self.kernel_slider.pack(padx=20)
        tk.Label(self.control_frame, text="Higher = Aggressively removes noise", font=("Helvetica", 8), bg="#4a5568", fg="gray").pack()

        # --- Run Button ---
        self.run_btn = ttk.Button(self.control_frame, text="Run Analysis", command=self.run_analysis)
        self.run_btn.pack(pady=30, padx=20, fill="x", ipady=5)

        # --- Status/Result Label ---
        self.result_frame = tk.Frame(self.control_frame, bg="#2d3748", bd=1, relief="solid")
        self.result_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.percent_label_title = tk.Label(self.result_frame, text="Percentage Changed:", font=("Helvetica", 12, "bold"), bg="#2d3748", fg="white")
        self.percent_label_title.pack(pady=(10,0))
        
        self.percent_label = tk.Label(self.result_frame, text="--%", font=("Helvetica", 24, "bold"), bg="#2d3748", fg="#38bdf8")
        self.percent_label.pack(pady=5)
        
        self.explanation_label = tk.Label(self.result_frame, text="Upload images to begin.", font=("Helvetica", 10), bg="#2d3748", fg="white", wraplength=250, justify="center")
        self.explanation_label.pack(pady=10, padx=10, fill="both", expand=True)

        # --- Matplotlib Plot Area ---
        self.fig = plt.figure(figsize=(10, 8), facecolor="#2d3748")
        self.fig.suptitle("Change Analysis Report", color="white", fontsize=16, fontweight='bold')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.create_placeholder_plots()

    def create_placeholder_plots(self):
        """Creates the initial 6-panel grid with placeholder text."""
        plt.clf() # Clear the figure
        
        ax1 = self.fig.add_subplot(231)
        ax1.set_title("Before (Normalized)", color="white")
        ax1.text(0.5, 0.5, "Upload 'Before' Image", color="gray", ha='center', va='center')
        ax1.set_facecolor("#4a5568")
        ax1.tick_params(colors='gray')

        ax2 = self.fig.add_subplot(232)
        ax2.set_title("After (Normalized)", color="white")
        ax2.text(0.5, 0.5, "Upload 'After' Image", color="gray", ha='center', va='center')
        ax2.set_facecolor("#4a5568")
        ax2.tick_params(colors='gray')
        
        ax3 = self.fig.add_subplot(233)
        ax3.set_title("Change Heatmap", color="white")
        ax3.set_facecolor("#4a5568")
        ax3.tick_params(colors='gray')
        
        ax4 = self.fig.add_subplot(234)
        ax4.set_title("Raw Difference", color="white")
        ax4.set_facecolor("#4a5568")
        ax4.tick_params(colors='gray')
        
        ax5 = self.fig.add_subplot(235)
        ax5.set_title("Final Change Mask", color="white")
        ax5.set_facecolor("#4a5568")
        ax5.tick_params(colors='gray')

        ax6 = self.fig.add_subplot(236)
        ax6.set_title("Change Overlay", color="white")
        ax6.set_facecolor("#4a5568")
        ax6.tick_params(colors='gray')
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for suptitle
        self.canvas.draw()

    def load_before(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            self.before_path = path
            self.before_label.config(text=os.path.basename(path))

    def load_after(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            self.after_path = path
            self.after_label.config(text=os.path.basename(path))

    def get_explanation(self, percent):
        """Generates a text explanation based on the percentage."""
        if percent < 5:
            return "Changes are minimal, likely due to minor lighting differences or seasonal vegetation changes. No major development detected."
        elif percent < 20:
            return "Noticeable changes detected. This could indicate small-scale construction, land clearing, or significant seasonal shifts."
        elif percent < 50:
            return "A significant portion of the area has undergone major changes, likely due to large-scale new construction, deforestation, or urban expansion."
        else:
            return "Massive changes detected. The landscape has been dramatically altered, indicating major development, deforestation, or natural disaster impact."

    def run_analysis(self):
        """Main analysis function, identical to the script's logic."""
        if not self.before_path or not self.after_path:
            self.explanation_label.config(text="Error: Please select both 'Before' and 'After' images.")
            return

        try:
            # --- 1. Load & Register Images ---
            img_before_raw = cv2.imread(self.before_path)
            img_after_raw = cv2.imread(self.after_path)

            if img_before_raw is None or img_after_raw is None:
                self.explanation_label.config(text="Error: Could not read one or both images.")
                return

            # --- Automated Registration (Resize) ---
            target_w = min(img_before_raw.shape[1], img_after_raw.shape[1])
            target_h = min(img_before_raw.shape[0], img_after_raw.shape[0])
            dsize = (target_w, target_h)
            
            img_before = cv2.resize(img_before_raw, dsize, interpolation=cv2.INTER_AREA)
            img_after = cv2.resize(img_after_raw, dsize, interpolation=cv2.INTER_AREA)

            # --- Get Settings ---
            THRESHOLD = self.threshold_slider.get()
            K_SIZE = self.kernel_slider.get()
            KERNEL_SIZE = (K_SIZE, K_SIZE)
            
            # --- Step 1: Grayscale and Denoise (Exp 1, 4) ---
            gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
            gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
            denoised_before = cv2.medianBlur(gray_before, 5)
            denoised_after = cv2.medianBlur(gray_after, 5)

            # --- Step 2: Normalize (Exp 5) ---
            norm_before = cv2.equalizeHist(denoised_before)
            norm_after = cv2.equalizeHist(denoised_after)
            
            # --- Step 3: Difference (Exp 3) ---
            diff_image = cv2.absdiff(norm_before, norm_after)

            # --- Step 4: Threshold (Exp 2) ---
            _ , thresh_mask = cv2.threshold(diff_image, THRESHOLD, 255, cv2.THRESH_BINARY)
            
            # --- Step 5: Refine Mask (Exp 7) ---
            kernel = np.ones(KERNEL_SIZE, np.uint8)
            opening_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            final_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # --- Step 6: Visualize & Quantify (Exp 8, Add-ons) ---
            
            # 1. Heatmap (Add-on)
            heatmap = cv2.applyColorMap(diff_image, cv2.COLORMAP_JET)
            
            # 2. Overlay (Exp 8, Add-on)
            overlay_image = img_after.copy()
            contours, _ = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 2) # BGR
            
            # 3. Percentage (Add-on)
            total_pixels = final_mask.shape[0] * final_mask.shape[1]
            change_pixels = cv2.countNonZero(final_mask)
            percent_changed = (change_pixels / total_pixels) * 100

            # --- 7. Update UI ---
            self.percent_label.config(text=f"{percent_changed:.2f}%")
            self.explanation_label.config(text=self.get_explanation(percent_changed))

            # --- 8. Update Matplotlib Figure ---
            self.fig.clear() # Clear the old plots
            self.fig.suptitle("Change Analysis Report", color="white", fontsize=16, fontweight='bold')

            # Panel 1: Before
            ax1 = self.fig.add_subplot(231)
            ax1.set_title('Before (Normalized)', color="white")
            ax1.imshow(norm_before, cmap='gray')
            ax1.tick_params(colors='gray')
            ax1.set_facecolor("#4a5568")

            # Panel 2: After
            ax2 = self.fig.add_subplot(232)
            ax2.set_title('After (Normalized)', color="white")
            ax2.imshow(norm_after, cmap='gray')
            ax2.tick_params(colors='gray')
            ax2.set_facecolor("#4a5568")

            # Panel 3: Heatmap
            ax3 = self.fig.add_subplot(233)
            ax3.set_title('Change Heatmap', color="white")
            ax3.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)) # <--- FIXED
            ax3.tick_params(colors='gray')
            ax3.set_facecolor("#4a5568")

            # Panel 4: Raw Difference
            ax4 = self.fig.add_subplot(234)
            ax4.set_title('Raw Difference', color="white")
            ax4.imshow(diff_image, cmap='gray')
            ax4.tick_params(colors='gray')
            ax4.set_facecolor("#4a5568")

            # Panel 5: Final Mask
            ax5 = self.fig.add_subplot(235)
            ax5.set_title('Final Change Mask', color="white")
            ax5.imshow(final_mask, cmap='gray')
            ax5.tick_params(colors='gray')
            ax5.set_facecolor("#4a5568")

            # Panel 6: Overlay
            ax6 = self.fig.add_subplot(236)
            ax6.set_title('Change Overlay', color="white")
            ax6.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)) # <--- FIXED
            ax6.tick_params(colors='gray')
            ax6.set_facecolor("#4a5568")

            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.canvas.draw()
            
        except Exception as e:
            self.explanation_label.config(text=f"An error occurred: {e}")
            print(f"Error: {e}")


# --- Main execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ChangeAnalyzerApp(root)
    root.mainloop()

