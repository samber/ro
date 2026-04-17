// Docusaurus client module — runs in the browser on every route

function posthog(): any | null {
  return typeof window !== 'undefined' ? (window as any).posthog ?? null : null;
}

// Sponsor link clicks (navbar + footer share one event, location field differentiates)
function initSponsorTracking() {
  document.addEventListener('click', (e) => {
    const ph = posthog();
    if (!ph) return;

    const link = (e.target as Element).closest('a[href*="sponsors/samber"]') as HTMLElement | null;
    if (!link) return;

    const location = link.closest('.navbar')
      ? 'navbar'
      : link.closest('footer')
      ? 'footer'
      : 'unknown';

    ph.capture('sponsor_clicked', { location });
  });
}

// Search queries — debounced to avoid capturing every keystroke
function initSearchTracking() {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  document.addEventListener('input', (e) => {
    const ph = posthog();
    if (!ph) return;

    const target = e.target as HTMLInputElement;
    if (!target.matches('.DocSearch-Input')) return;

    const query = target.value.trim();
    if (!query) return;

    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => {
      ph.capture('search_query', { query });
    }, 600);
  });
}

if (typeof window !== 'undefined') {
  initSponsorTracking();
  initSearchTracking();
}

// Required export for Docusaurus client modules
export function onRouteDidUpdate() {}
