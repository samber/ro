import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'samber/ro - Reactive Streams for Go',
  tagline: 'Streams and reactive programming for Go',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: {
      removeLegacyPostBuildHeadAttribute: true,
      useCssCascadeLayers: true,
    },
    faster: {
        swcJsLoader: true,
        swcJsMinimizer: true,
        swcHtmlMinimizer: true,
        lightningCssMinimizer: true,
        rspackBundler: true,
        rspackPersistentCache: true,
        ssgWorkerThreads: true,
        mdxCrossCompilerCache: true,
    },
  },
    storage: {
        type: 'localStorage',
        namespace: true,
    },

    // Set the production url of your site here
  url: 'https://ro.samber.dev',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'samber', // Usually your GitHub org/user name.
  projectName: 'ro', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'throw',
  onBrokenAnchors: 'throw',

  markdown: {
    anchors: {
      maintainCase: true,
    },
    mermaid: true,
  },

  // Storage configuration for better performance
  staticDirectories: ['static'],

  // Optional: Enable hash router for offline support (experimental)
  // Uncomment if you need offline browsing capability
  // router: 'hash',

    // Future-proofing configurations
  clientModules: [
    require.resolve('./src/theme/prism-include-languages.js'),
    require.resolve('./src/analytics.ts'),
  ],

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

    headTags: [
        // SEO
        {
            tagName: 'meta',
            attributes: {
                name: 'msvalidate.01',
                content: '4576E3F85783A82149A0DB35A150F7EB',
            },
        },
        {
        tagName: 'script',
        attributes: {
            'async': 'true',
            'src': 'https://analytics.ahrefs.com/analytics.js',
            'data-key': 'ZlVVDleFCGZPB8Nd2KkKrw'
        }
    },
    // NOTE: no dns-prefetch/preconnect to fonts.googleapis.com/fonts.gstatic.com
    // here — the site doesn't load any Google Font, so those hints only
    // opened two unused connections on every page load.
    // NOTE: no `keywords` meta either — Google has ignored it for SEO
    // ranking for years, and it only added dead weight to every page.
    // og:image, twitter:card, twitter:image and og:locale are NOT declared
    // here: Docusaurus already generates them from `themeConfig.image` and
    // `i18n.defaultLocale` below. Duplicating them produced two conflicting
    // <meta property="og:locale"> tags in the rendered HTML (this file said
    // en_US, Docusaurus's own tag said en).
    {
      tagName: 'meta',
      attributes: {
        name: 'twitter:creator',
        content: '@samuelberthe',
      },
    },
    // twitter:site complements twitter:creator for card attribution
    {
      tagName: 'meta',
      attributes: {
        name: 'twitter:site',
        content: '@samuelberthe',
      },
    },
    // og:site_name provides branding context in social share cards
    {
      tagName: 'meta',
      attributes: {
        property: 'og:site_name',
        content: 'samber/ro',
      },
    },
        // NOTE: do not add a global <link rel="canonical"> here.
        // Docusaurus generates correct per-page canonical tags from the `url` field above.
        // A global canonical in headTags applies to ALL pages and conflicts with those.
    ],

    customFields: {
        sponsors: [
      {
        name: 'DBOS',
        url: 'https://www.dbos.dev/?utm_campaign=gh-smbr',
        title: 'DBOS - Durable workflow orchestration library for Go',
        logo_light: '/img/sponsors/dbos-black.png',
        logo_dark: '/img/sponsors/dbos-white.png',
      },
    ],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
          'https://github.com/samber/ro/tree/main/docs/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          // Enhanced docs features from 3.8+
          breadcrumbs: true,
          sidebarCollapsed: false,
          numberPrefixParser: false,
          // Enable admonitions
          admonitions: {
            keywords: ['note', 'tip', 'info', 'danger', 'warning'],
            extendDefaults: true,
          },
          // Enhanced markdown features
          remarkPlugins: [],
          rehypePlugins: [],
        },
        // No blog/ directory exists — the sidebar's "Blog" entry links to
        // Substack instead. Without this, Docusaurus's classic preset
        // still enables its default blog plugin and publishes an empty,
        // indexed /blog page.
        blog: false,
          sitemap: {
          lastmod: 'date',
          changefreq: 'weekly',
          priority: 0.7,
          ignorePatterns: ['/tags/**', '/search'],
          filename: 'sitemap.xml',
          // Enhanced sitemap features from 3.8+
          createSitemapItems: async (params) => {
            const {defaultCreateSitemapItems, ...rest} = params ;
            const items = await defaultCreateSitemapItems(rest);
            // Add custom priority for specific pages
            return items.map((item) => {
              if (item.url.includes('/docs/getting-started')) {
                return {...item, priority: 1.0};
              }
              if (item.url.includes('/docs/')) {
                return {...item, priority: 0.8};
              }
              return item;
            });
          },
        },
        theme: {
          customCss: './src/css/custom.css',
        },
        gtag: {
          trackingID: 'G-5DCQT10D37',
          anonymizeIP: false,
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/cover.jpg',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },

    // Mermaid configuration
    mermaid: {
      theme: {light: 'neutral', dark: 'dark'},
      options: {
        maxTextSize: 50000,
      },
    },

      // Enhanced metadata
    metadata: [
      {property: 'og:type', content: 'website'},
      // Fallback description for pages that don't set their own
      {name: 'description', content: 'Reactive programming for Go using generics. An implementation of the ReactiveX spec with Observables, Operators, and Subjects for building event-driven and asynchronous applications.'},
    ],

    navbar: {
      title: '🌊 samber/ro',
      logo: {
        alt: 'ro - Reactive programming for Go',
        src: 'img/icon.png',
        width: 32,
        height: 32,
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docSidebar',
          position: 'left',
          label: 'Doc',
        },
        {
          to: 'examples',
          label: 'Examples',
          position: 'left',
        },
        {
          to: 'https://pkg.go.dev/github.com/samber/ro',
          label: 'GoDoc',
          position: 'left',
        },
        {
          to: 'community',
          label: 'Community',
          position: 'left',
        },
        {
          to: 'https://github.com/samber/ro/releases',
          label: 'Changelog',
          position: 'right',
        },
        {
          to: 'https://github.com/sponsors/samber',
          label: '💖 Sponsor',
          position: 'right',
        },
        {
          href: 'https://github.com/samber/ro',
          // label: 'GitHub',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
        {
          type: 'search',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Project',
          items: [
            {
              label: 'Documentation',
              to: '/docs/getting-started',
            },
            {
              label: 'Changelog',
              to: 'https://github.com/samber/ro/releases',
            },
            {
              label: 'Godoc',
              to: 'https://pkg.go.dev/github.com/samber/ro',
            },
            {
              label: 'License',
              to: 'https://github.com/samber/ro/blob/main/LICENSE',
            },
            {
              label: '💖 Sponsor',
              to: 'https://github.com/sponsors/samber',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'New issue',
              to: 'https://github.com/samber/ro/issues',
            },
            {
              label: 'GitHub',
              to: 'https://github.com/samber/ro',
            },
            {
              label: 'Stack Overflow',
              to: 'https://stackoverflow.com/search?q=samber+ro',
            },
            {
              label: 'Twitter',
              to: 'https://twitter.com/samuelberthe',
            },
            {
              label: 'Substack',
              to: 'https://samuelberthe.substack.com',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} ro.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      defaultLanguage: 'go',
      additionalLanguages: ['bash', 'diff', 'json', 'yaml', 'go'],
      magicComments: [
        {
          className: 'theme-code-block-highlighted-line',
          line: 'highlight-next-line',
          block: {start: 'highlight-start', end: 'highlight-end'},
        },
        {
          className: 'code-block-error-line',
          line: 'error-next-line',
          block: {start: 'error-start', end: 'error-end'},
        },
      ],
    },
    algolia: {
      appId: 'XHFWP01VWP',
      // bearer:disable javascript_lang_hardcoded_secret
      apiKey: '1a422e992fcddad0f84d082a9620040c',
      externalUrlRegex: 'ro\\.samber\\.dev',
      indexName: 'ro.samber.dev',
      contextualSearch: true,
      searchParameters: {
        // facetFilters: ['type:lvl1'],
      },
      searchPagePath: 'search',
      // Enhanced search features from 3.8+
      insights: true,
    },
  } satisfies Preset.ThemeConfig,

  themes: ['@docusaurus/theme-mermaid'],

    plugins: [
        [
        "posthog-docusaurus",
        {
            apiKey: "phc_z838Jn9KD8Da3ue9Z7htxyw3QEvTwm9tsHbHebhCDnSd",
            appHost: "https://hogpost.samber.dev",
            enableInDevelopment: false, // optional,
            disableSessionRecording: true,
        },
    ],
        // Add ideal image plugin for better image optimization
        [
      '@docusaurus/plugin-ideal-image',
      {
        quality: 70,
        max: 1030,
        min: 640,
        steps: 2,
        disableInDev: false,
      },
    ],
    [
      'vercel-analytics',
      {
        debug: true,
        mode: 'auto',
      },
    ],
    // Custom plugin to generate helper category pages from data markdown
    [
      require.resolve('./plugins/helpers-pages'),
      {},
    ],
  ],
};

export default config;
