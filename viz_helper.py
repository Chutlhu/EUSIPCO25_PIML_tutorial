import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


# Plot/animate
def animate_wave_1D_periodic(
    u_tx, 
    x, 
    times, 
    title='',
    L=1,
    interval=25, 
    u_ref=None, 
    show_err=False,
    save_path=None,
    mode='js'
    ):
    
    fig, ax = plt.subplots(figsize=(7,3))
    n_time, n_space = u_tx.shape
    if u_ref is not None:
        (line_ref,) = ax.plot(x, u_ref[0], 'C0--', lw=2, label='Analytic')
        (line,) = ax.plot(x, u_tx[0], 'C1', lw=2, label='Solution')
    if u_ref is not None and show_err:
        err = np.abs(u_tx[0]-u_ref[0])
        (line_err,) = ax.plot(x, err, 'k--', lw=1, label='Error')
    
    # set the plots
    vmin, vmax = -1.1*np.max(np.abs(u_tx[0])), 1.1*np.max(np.abs(u_tx[0]))
    ax.set_xlim(0, L)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"{title}, t=0.000 s")
    ax.legend(loc='upper right')

    def update(i):
        if u_ref is not None:
            line_ref.set_ydata(u_ref[i])
            line.set_ydata(u_tx[i])
        if u_ref is not None and show_err:
         line_err.set_ydata(np.abs(u_tx[i]-u_ref[i]))
        # ax.scatter([0,L], [u_tx[0,0], u_tx[0,-1]], color='k', label='B.C.')
        ax.set_title(f"{title}, t={times[i]:.3f} s")
        return (line,)

    anim = FuncAnimation(
        fig, update, frames=n_time, interval=interval,
        blit=True, cache_frame_data=False
    )

    # output: js widget (default), HTML5 video, or GIF
    if mode == "html":
        from IPython.display import HTML
        html = HTML(anim.to_html5_video())   # smoother playback for large Nt
        plt.close(fig)
        if save_path:  # optional save mp4
            try: anim.save(save_path, writer='ffmpeg', fps=max(1,int(1000/interval)))
            except Exception as e: print(f"[warn] save mp4 failed: {e}")
        return html
    elif mode == "gif":
        if save_path is None: save_path = "anim.gif"
        try: anim.save(save_path, writer='pillow', fps=max(1,int(1000/interval)))
        except Exception as e: print(f"[warn] save gif failed: {e}")
        plt.close(fig)
        return HTML(f'<img src="{save_path}">')
    else:
        from matplotlib import rc
        rc('animation','jshtml')
        plt.close(fig)   # suppress extra static plot
        return anim
    