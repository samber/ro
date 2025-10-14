import React from 'react';

import styles from './examples.module.css';
import classnames from 'classnames';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

function Examples() {
  const context = useDocusaurusContext();

  return (
    <Layout title="Examples" description="Projects implementing samber/ro">
      <header className="hero">
        <div className="container text--center">
          <h1>Examples and templates</h1>
          <div className="hero--subtitle">
            Here are a few demo projects implementing `samber/ro`.
          </div>
          <img className={styles.headerImg} src="/img/go-templates.png" />
        </div>
      </header>
      <main>
        <div className="container">
          <div className="row margin-vert--lg">
            <div className="col text--center padding-vert--md">
              <div className="card">
                <div className="card__header">
                  <i className={classnames(styles.icon)}>🚀</i>
                </div>
                <div className="card__body">
                  <p>samber/ro &gt; examples</p>
                </div>
                <div className="card__footer">
                  <Link to="https://github.com/samber/ro/tree/master/examples" className="button button--outline button--primary button--block">See examples</Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default Examples;