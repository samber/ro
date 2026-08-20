import React from 'react';

import styles from './examples.module.css';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

type ExampleItem = {
  title: string;
  description: string;
  to: string;
};

const EXAMPLES: ExampleItem[] = [
  {
    title: 'Stock price enrichment',
    description: 'A websocket client that enriches incoming price ticks with reference data.',
    to: 'https://github.com/samber/ro/tree/main/examples/stock-price-enrichment',
  },
  {
    title: 'Distributed WebSocket gateway',
    description: 'Fans a single upstream feed out to many WebSocket clients.',
    to: 'https://github.com/samber/ro/tree/main/examples/distributed-websocket-gateway',
  },
  {
    title: 'Parallel API requests',
    description: 'Concurrent HTTP requests over a bounded pipeline.',
    to: 'https://github.com/samber/ro/tree/main/examples/parallel-api-requests',
  },
  {
    title: 'SQL to CSV',
    description: 'Streams database query results straight into a CSV file.',
    to: 'https://github.com/samber/ro/tree/main/examples/sql-to-csv',
  },
  {
    title: 'ICS to CSV',
    description: 'Parses an ICS calendar and converts its events to CSV.',
    to: 'https://github.com/samber/ro/tree/main/examples/ics-to-csv',
  },
  {
    title: 'Connectable observables',
    description: 'Shares one execution across multiple subscribers with a hot observable.',
    to: 'https://github.com/samber/ro/tree/main/examples/connectable',
  },
];

const EE_EXAMPLES: ExampleItem[] = [
  {
    title: 'OpenTelemetry logs',
    description: 'Enterprise Edition — piping pipeline logs to OpenTelemetry.',
    to: 'https://github.com/samber/ro/tree/main/examples/ee-otel-log',
  },
  {
    title: 'OpenTelemetry metrics',
    description: 'Enterprise Edition — exposing pipeline metrics via OpenTelemetry.',
    to: 'https://github.com/samber/ro/tree/main/examples/ee-otel-metrics',
  },
  {
    title: 'OpenTelemetry tracing',
    description: 'Enterprise Edition — tracing a pipeline end to end with OpenTelemetry.',
    to: 'https://github.com/samber/ro/tree/main/examples/ee-otel-tracing',
  },
  {
    title: 'Prometheus',
    description: 'Enterprise Edition — exporting pipeline metrics to Prometheus.',
    to: 'https://github.com/samber/ro/tree/main/examples/ee-prometheus',
  },
];

function ExampleCard({title, description, to}: ExampleItem) {
  return (
    <div className="col col--4 margin-bottom--lg">
      <Link to={to} className={styles.exampleCard}>
        <div className="card">
          <div className="card__header">
            <h3>{title}</h3>
          </div>
          <div className="card__body">
            <p>{description}</p>
          </div>
        </div>
      </Link>
    </div>
  );
}

function Examples() {
  return (
    <Layout title="Examples" description="Working example projects built with samber/ro">
      <header className="hero">
        <div className="container text--center">
          <h1>Examples and templates</h1>
          <div className="hero--subtitle">
            Production-shaped projects built with samber/ro: websocket ingestion, parallel HTTP fan-out, SQL and calendar pipelines.
          </div>
          <img className={styles.headerImg} src="/img/go-templates-optimized.png" alt="Go project templates" width={300} height={300} />
        </div>
      </header>
      <main>
        <div className="container margin-vert--lg">
          <div className="row">
            {EXAMPLES.map((example) => (
              <ExampleCard key={example.to} {...example} />
            ))}
          </div>

          <h2>Enterprise Edition</h2>
          <p>
            These examples use plugins under a separate license — see{' '}
            <Link to="https://github.com/samber/ro/blob/main/ee/README.md">ee/README.md</Link>.
          </p>
          <div className="row">
            {EE_EXAMPLES.map((example) => (
              <ExampleCard key={example.to} {...example} />
            ))}
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default Examples;