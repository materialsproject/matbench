/* Keep Plotly inside its own scroll region and make the wrapping legend usable
 * with a mouse, keyboard, or touch. The score arrays are never changed here. */
(() => {
  const initializeCharts = () => {
    document.querySelectorAll('.matbench-chart').forEach((section) => {
      if (section.dataset.chartReady === 'true') return;
      const canvas = section.querySelector('.matbench-chart-canvas');
      const region = section.querySelector('.matbench-chart-scroll');
      const plot = canvas && canvas.querySelector('.js-plotly-plot');
      if (!plot || !region) return;
      section.dataset.chartReady = 'true';
      section.querySelectorAll('.matbench-chart-key').forEach((button) => {
        button.addEventListener('click', () => {
          if (!window.Plotly || !plot.data) return;
          const index = Number(button.dataset.trace);
          const visible = plot.data[index].visible;
          const show = visible === false || visible === 'legendonly';
          window.Plotly.restyle(plot, {visible: show ? true : 'legendonly'}, [index]);
          button.setAttribute('aria-pressed', String(show));
        });
      });
      let previousWidth = 0;
      const resize = () => {
        section.dataset.overflow = String(region.scrollWidth > region.clientWidth + 1);
        const width = canvas.getBoundingClientRect().width;
        if (width === previousWidth || !window.Plotly || !plot._fullLayout) return;
        previousWidth = width;
        window.requestAnimationFrame(() => window.Plotly.Plots.resize(plot));
      };
      if (window.ResizeObserver) {
        const observer = new ResizeObserver(resize);
        observer.observe(region);
        observer.observe(canvas);
      } else {
        window.addEventListener('resize', resize);
      }
      resize();
    });
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeCharts);
  } else {
    initializeCharts();
  }
  // Also support Material's optional instant navigation if enabled later.
  if (typeof document$ !== 'undefined') document$.subscribe(initializeCharts);
})();
