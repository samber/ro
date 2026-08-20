import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';
import styles from './index.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Streams beyond Go channels',
    Svg: require('@site/static/img/tram.svg').default,
    description: (
      <>
        You already know channels and goroutines. <code>ro</code> adds
        what's missing: a vocabulary of composable operators, so you
        describe what a stream does instead of coding the concurrency
        around it by hand.
      </>
    ),
  },
  {
    title: 'Transformation chaining',
    Svg: require('@site/static/img/street-sign.svg').default,
    description: (
      <>
        Compose <code>Map</code>, <code>Filter</code>, and 200+ other
        operators into a single fluent pipeline. Every step is fully
        generic-typed, so a mismatched transformation is a compile error,
        not a runtime surprise.
      </>
    ),
  },
  {
    title: 'A small core, a powerful developer experience',
    Svg: require('@site/static/img/drawing.svg').default,
    description: (
      <>
        A minimal yet expressive API that gives you full control over your
        reactive pipelines, without sacrificing clarity or performance. The
        core package depends only on <code>samber/lo</code> — everything
        else (JSON, HTTP, rate limiting, structured logging) lives in
        opt-in plugin modules you import only when you need them.
      </>
    ),
  },
];

const HERO_EXAMPLE = `observable := ro.Pipe[int64, string](
    ro.RangeWithInterval(0, 5, 1*time.Second),
    ro.Filter(func(x int64) bool {
        return x%2 == 0
    }),
    ro.Map(func(x int64) string {
        return fmt.Sprintf("even-%d", x)
    }),
)

observable.Subscribe(ro.NewObserver(
    func(s string) { fmt.Println(s) },
    func(err error) { fmt.Println(err.Error()) },
    func() { fmt.Println("Completed!") },
))
// "even-0"
// "even-2"
// "even-4"
// "Completed!"`;

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

function Badges(): ReactNode {
  return (
    <div className={styles.badges}>
      <a href="https://github.com/samber/ro/releases">
        <img src="https://img.shields.io/github/tag/samber/ro.svg" alt="Latest tag" loading="lazy" />
      </a>
      <a href="https://pkg.go.dev/github.com/samber/ro">
        <img src="https://godoc.org/github.com/samber/ro?status.svg" alt="GoDoc" loading="lazy" />
      </a>
      <a href="https://github.com/samber/ro/actions/workflows/test.yml">
        <img src="https://github.com/samber/ro/actions/workflows/test.yml/badge.svg" alt="Build status" loading="lazy" />
      </a>
      <a href="https://goreportcard.com/report/github.com/samber/ro">
        <img src="https://goreportcard.com/badge/github.com/samber/ro" alt="Go report card" loading="lazy" />
      </a>
      <a href="https://github.com/samber/ro/stargazers">
        <img src="https://img.shields.io/github/stars/samber/ro?style=social" alt="GitHub stars" loading="lazy" />
      </a>
    </div>
  );
}

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">
          Compose async, event-driven pipelines in Go — Observables, 200+
          type-safe operators, and a small dependency-free core.
        </p>
        <Badges />
        <div className={styles.heroCode}>
          <CodeBlock language="go" title="main.go">
            {HERO_EXAMPLE}
          </CodeBlock>
        </div>
        <div className={clsx(styles.buttons, 'margin-top--md')}>
          <Link className="button button--primary button--lg" to="/docs/getting-started">
            Getting started — 5 min ⏱️
          </Link>
          <Link className="button button--secondary button--lg" to="/docs/about">
            Why reactive in Go?
          </Link>
          <Link className="button button--secondary button--lg" to="https://github.com/samber/ro">
            GitHub
          </Link>
        </div>
        <p className={styles.installHint}>
          <code>go get -u github.com/samber/ro</code>
        </p>
      </div>
    </header>
  );
}

function WhyNotChannels(): ReactNode {
  return (
    <section className={styles.why}>
      <div className="container">
        <div className="row">
          <div className="col col--8 col--offset-2 text--center">
            <Heading as="h2">Why not just channels?</Heading>
            <p>
              You can build any of this with raw channels and goroutines —{' '}
              <code>ro</code>'s own operators are implemented that way
              underneath. The difference is that you write the fan-out,
              retry, debounce, and cleanup logic once, as a reusable
              operator, instead of re-deriving it in every service. If a
              plain channel already does the job, keep the channel — see{' '}
              <Link to="/docs/comparison/channels-vs-ro">channels vs ro</Link>{' '}
              for a side-by-side comparison.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

type ExploreCard = {
  title: string;
  description: string;
  to: string;
};

const EXPLORE_CARDS: ExploreCard[] = [
  {
    title: '👷 Operators',
    description: '200+ creation, transformation, filtering, and combining operators.',
    to: '/docs/operator',
  },
  {
    title: '🔍 Plugins',
    description: 'JSON, CSV, HTTP, rate limiting, structured logging, and more — opt-in modules.',
    to: '/docs/plugins',
  },
  {
    title: '📊 Comparisons',
    description: 'How ro relates to channels, iter, samber/lo, RxGo, and RxJS.',
    to: '/docs/comparison',
  },
];

function ExploreSection(): ReactNode {
  return (
    <section className={styles.explore}>
      <div className="container">
        <div className="row">
          {EXPLORE_CARDS.map((card) => (
            <div key={card.to} className="col col--4 margin-bottom--lg">
              <Link to={card.to} className={clsx('card', styles.exploreCard)}>
                <div className="card__header">
                  <Heading as="h3">{card.title}</Heading>
                </div>
                <div className="card__body">
                  <p>{card.description}</p>
                </div>
              </Link>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): JSX.Element {
  return (
    <Layout
      title="Type-safe async pipelines for Go"
      description="ro is a Go implementation of the ReactiveX spec: Observables, Observers, and 200+ type-safe operators, built on Go 1.18+ generics, for event-driven and asynchronous applications.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <WhyNotChannels />
        <ExploreSection />
      </main>
    </Layout>
  );
}
