import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import sys

def create_dashboard(filename):
    df = pd.read_csv(filename)
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a0a')
    
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    total_time = df['elapsed_seconds'].max()
    minutes = total_time // 60
    seconds = total_time % 60
    
    focus_counts = df['focus_state'].value_counts()
    focused_pct = (focus_counts.get('focused', 0) / len(df) * 100)
    distracted_pct = (focus_counts.get('distracted', 0) / len(df) * 100)
    drowsy_pct = (focus_counts.get('drowsy', 0) / len(df) * 100)
    
    title = fig.suptitle('FOCUSTRACK SESSION DASHBOARD', 
                         fontsize=24, fontweight='bold', color='white', y=0.98)
    
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.set_facecolor('#1a1a1a')
    
    color_map = {
        'focused': '#00ff41',
        'distracted': '#ff0054',
        'drowsy': '#9d4edd',
        'neutral': '#ffd60a'
    }
    
    for i, row in df.iterrows():
        color = color_map.get(row['focus_state'], '#ffffff')
        ax_timeline.barh(0, width=2, left=row['elapsed_seconds'], 
                        height=1, color=color, edgecolor='none')
    
    ax_timeline.set_xlim(0, total_time)
    ax_timeline.set_ylim(-0.6, 0.6)
    ax_timeline.set_xlabel('Time (seconds)', fontsize=11, color='white', fontweight='bold')
    ax_timeline.set_title('FOCUS TIMELINE', fontsize=13, color='white', fontweight='bold', pad=10)
    ax_timeline.set_yticks([])
    ax_timeline.tick_params(colors='white')
    ax_timeline.spines['top'].set_visible(False)
    ax_timeline.spines['right'].set_visible(False)
    ax_timeline.spines['left'].set_visible(False)
    ax_timeline.spines['bottom'].set_color('white')
    
    legend_elements = [
        mpatches.Patch(facecolor='#00ff41', label='Focused'),
        mpatches.Patch(facecolor='#ff0054', label='Distracted'),
        mpatches.Patch(facecolor='#9d4edd', label='Drowsy')
    ]
    ax_timeline.legend(handles=legend_elements, loc='upper right', 
                      fontsize=9, facecolor='#2a2a2a', edgecolor='white')
    
    ax_pie = fig.add_subplot(gs[1, 0])
    ax_pie.set_facecolor('#1a1a1a')
    
    pie_colors = {
        'focused': '#00ff41',
        'distracted': '#ff0054',
        'drowsy': '#9d4edd',
        'neutral': '#ffd60a'
    }
    
    pie_data = [focused_pct, distracted_pct, drowsy_pct]
    pie_labels = ['Focused', 'Distracted', 'Drowsy']
    pie_color_list = ['#00ff41', '#ff0054', '#9d4edd']
    
    wedges, texts, autotexts = ax_pie.pie(
        pie_data,
        labels=pie_labels,
        autopct='%1.1f%%',
        colors=pie_color_list,
        startangle=90,
        textprops={'color': 'white', 'fontsize': 10, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    ax_pie.set_title('FOCUS BREAKDOWN', fontsize=13, color='white', fontweight='bold', pad=10)
    
    ax_emotions = fig.add_subplot(gs[1, 1])
    ax_emotions.set_facecolor('#1a1a1a')
    
    emotion_counts = df['emotion'].value_counts().head(5)
    
    emotion_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7']
    
    bars = ax_emotions.barh(emotion_counts.index, emotion_counts.values, 
                           color=emotion_colors[:len(emotion_counts)],
                           edgecolor='white', linewidth=1.5)
    
    for i, v in enumerate(emotion_counts.values):
        ax_emotions.text(v + 0.2, i, str(v), 
                        va='center', fontsize=10, fontweight='bold', color='white')
    
    ax_emotions.set_xlabel('Count', fontsize=10, color='white', fontweight='bold')
    ax_emotions.set_title('TOP EMOTIONS', fontsize=13, color='white', fontweight='bold', pad=10)
    ax_emotions.tick_params(colors='white')
    ax_emotions.spines['top'].set_visible(False)
    ax_emotions.spines['right'].set_visible(False)
    ax_emotions.spines['bottom'].set_color('white')
    ax_emotions.spines['left'].set_color('white')
    
    ax_blinks = fig.add_subplot(gs[1, 2])
    ax_blinks.set_facecolor('#1a1a1a')
    
    ax_blinks.plot(df['elapsed_seconds'], df['blinks_per_min'], 
                  color='#00d2ff', linewidth=2.5, marker='o', markersize=5)
    
    ax_blinks.axhline(y=8, color='#9d4edd', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_blinks.axhline(y=25, color='#ff0054', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_blinks.fill_between(df['elapsed_seconds'], 8, 25, alpha=0.15, color='#00ff41')
    
    ax_blinks.set_xlabel('Time (seconds)', fontsize=10, color='white', fontweight='bold')
    ax_blinks.set_ylabel('Blinks/Min', fontsize=10, color='white', fontweight='bold')
    ax_blinks.set_title('BLINK RATE', fontsize=13, color='white', fontweight='bold', pad=10)
    ax_blinks.tick_params(colors='white')
    ax_blinks.grid(True, alpha=0.2, color='white')
    ax_blinks.spines['top'].set_visible(False)
    ax_blinks.spines['right'].set_visible(False)
    ax_blinks.spines['bottom'].set_color('white')
    ax_blinks.spines['left'].set_color('white')
    
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    avg_blinks = df['blinks_per_min'].mean()
    top_emotion = df['emotion'].value_counts().index[0]
    productivity = int(focused_pct)
    
    stats_boxes = [
        {
            'title': 'DURATION',
            'value': f'{int(minutes)}m {int(seconds)}s',
            'color': '#00d2ff'
        },
        {
            'title': 'PRODUCTIVITY',
            'value': f'{productivity}%',
            'color': '#00ff41' if productivity >= 50 else '#ff0054'
        },
        {
            'title': 'FOCUSED TIME',
            'value': f'{focused_pct:.1f}%',
            'color': '#00ff41'
        },
        {
            'title': 'DISTRACTED',
            'value': f'{distracted_pct:.1f}%',
            'color': '#ff0054'
        },
        {
            'title': 'DROWSY',
            'value': f'{drowsy_pct:.1f}%',
            'color': '#9d4edd'
        },
        {
            'title': 'AVG BLINKS',
            'value': f'{avg_blinks:.1f}/min',
            'color': '#00d2ff'
        },
        {
            'title': 'TOP EMOTION',
            'value': top_emotion.upper(),
            'color': '#ffd60a'
        },
        {
            'title': 'DATA POINTS',
            'value': str(len(df)),
            'color': '#a29bfe'
        }
    ]
    
    box_width = 1.0 / len(stats_boxes)
    
    for i, box in enumerate(stats_boxes):
        x_pos = i * box_width + box_width / 2
        
        rect = mpatches.FancyBboxPatch(
            (i * box_width + 0.01, 0.1),
            box_width - 0.02,
            0.8,
            boxstyle="round,pad=0.02",
            facecolor='#1a1a1a',
            edgecolor=box['color'],
            linewidth=2.5,
            transform=ax_stats.transAxes
        )
        ax_stats.add_patch(rect)
        
        ax_stats.text(x_pos, 0.7, box['title'],
                     ha='center', va='center',
                     fontsize=9, fontweight='bold',
                     color='#888888',
                     transform=ax_stats.transAxes)
        
        ax_stats.text(x_pos, 0.35, box['value'],
                     ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     color=box['color'],
                     transform=ax_stats.transAxes)
    
    output_name = filename.replace('.csv', '_dashboard.png')
    plt.savefig(output_name, dpi=300, bbox_inches='tight', 
                facecolor='#0a0a0a', edgecolor='none')
    
    print(f"\nDashboard saved: {output_name}")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dashboard.py <session_file.csv>")
    else:
        create_dashboard(sys.argv[1])