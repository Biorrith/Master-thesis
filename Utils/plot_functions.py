import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from sksurv.datasets import get_x_y

font_path = '../Old_Standard_TT'
font_path_regular = f'{font_path}/OldStandardTT-Regular.ttf'
font_path_bold = f'{font_path}/OldStandardTT-Bold.ttf'
font_path_italic = f'{font_path}/OldStandardTT-Italic.ttf'

# Specify the direct path to your font file
# font_path_regular = '/home/diego/.conda/envs/diego/fonts/OldStandardTT-Regular.ttf'
# font_path_bold = '/home/diego/.conda/envs/diego/fonts/OldStandardTT-Bold.ttf'
# font_path_italic = '/home/diego/.conda/envs/diego/fonts/OldStandardTT-Italic.ttf'
# Create a FontProperties object with the full path to the font file
prop_regular = FontProperties(fname=font_path_regular)
prop_bold = FontProperties(fname=font_path_bold)
prop_italic = FontProperties(fname=font_path_italic)
my_colors = ['#B0DAFF', '#FFB085']


def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def get_full_color_gradient(colors, n):
    """
    Given an array of hex colors, returns a color gradient
    with n colors smoothly transitioning across all given colors.
    """
    assert len(colors) > 1 and n > len(colors)
    k = len(colors)  # Number of colors
    segments = k - 1  # Number of transitions between colors
    
    # Calculate number of steps for each segment, handling possible uneven division
    steps_per_segment = [n // segments + (1 if i < n % segments else 0) for i in range(segments)]

    # Initialize the list to hold all gradient colors
    gradient = []

    # Generate gradient for each segment
    for i in range(segments):
        c1 = np.array(hex_to_RGB(colors[i])) / 255
        c2 = np.array(hex_to_RGB(colors[i+1])) / 255
        mix_pcts = [x / (steps_per_segment[i] - 1) for x in range(steps_per_segment[i])]
        rgb_segment = [((1 - mix) * c1 + mix * c2) for mix in mix_pcts]
        gradient.extend(rgb_segment if i == 0 else rgb_segment[1:])

    # Convert RGB colors to hex format
    hex_gradient = ["#" + "".join([format(int(round(val * 255)), "02x") for val in color]) for color in gradient]
    return hex_gradient


def plot_errorbar(coef, ci_lower, ci_upper, path, title, yticks):
    # Create an array for the positions on the y-axis
    y_pos = np.arange(len(coef))

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot the points with error bars
    ax.errorbar(coef, y_pos, xerr=[coef - ci_lower, ci_upper - coef], fmt='o', color='#91C6FF', ecolor='#FF8F61', capsize=6, zorder=3, markersize=12, linewidth=2.5, capthick=2.5 )

    # Add y-ticks and labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(yticks.index, fontsize=24, fontproperties=prop_regular)

    # Labels and title
    ax.set_xlabel('Coefficient', fontsize=28, fontproperties=prop_bold)
    ax.set_title(f'Coefficients and 95% Confidence Intervals of {title} Model', fontproperties=prop_bold, fontsize=32)

    # Set custom ticks for x-axis (optional, adjust as needed)
    # ax.set_xticks(np.arange(-3, 4, 1))
    ax.tick_params(axis='x', labelsize=2)
    for label in ax.get_xticklabels():
        label.set_fontproperties(prop_regular)
        label.set_fontsize(12)


    # Grid for better readability
    ax.grid(axis='y', zorder=0)

    # Add the grid lines for the y-axis
    plt.tight_layout()
    plt.savefig(f"./results/{path}.pdf")
    plt.grid(True)


def plot_bar_difference(model, df_test_inp, label, path):
    df_test = df_test_inp.copy()
    df_test = df_test[df_test['Event'] ==1]
  
    x_test, y_test = get_x_y(df_test, attr_labels=["Event", "M"], pos_label=1)
    pred_surv_funcs = model.predict_survival_function(x_test)
    pred_times = []
    for fn in pred_surv_funcs:
    # Check if there are any points where fn.y <= 0.5
        # print(fn.x[fn.y <= 0.5])
        below_0_5 = fn.x[fn.y <= 0.5]
        if below_0_5.size > 0:
            pred_times.append(below_0_5[0])
        else:
            # Use the maximum time if no probability drops below 0.5
            pred_times.append(fn.x[-1])

    time_diff = pred_times - y_test['M']
    bins = np.arange(-48, 48, 6)
    counts, _ = np.histogram(time_diff, bins=bins)

    plt.figure(figsize=(12, 8))

    plt.bar(bins[:-1], counts, color=get_color_gradient(my_colors[0], my_colors[1], len(bins)), zorder=3, width=3)
    plt.title(f'Difference Between {label} Model Prediction and Actual Survival Function', fontproperties=prop_bold, fontsize=25)
    plt.xlabel('Count', fontsize=25, fontproperties=prop_bold)
    plt.ylabel('Difference(months)', fontsize=25, fontproperties=prop_bold, labelpad=25)
    
    plt.xticks(bins, fontproperties=prop_regular, fontsize=15)
    plt.yticks(np.arange(0,counts.max()+1),fontproperties=prop_regular, fontsize=15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"./results/{path}_barplot.pdf")
    plt.show()