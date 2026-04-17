import React from 'react';
import CopyButton from '@theme-original/CodeBlock/CopyButton';

type Props = {
  code: string;
  [key: string]: unknown;
};

export default function CopyButtonWrapper(props: Props): React.ReactElement {
  const handleCopy = () => {
    if (typeof window !== 'undefined' && (window as any).posthog) {
      (window as any).posthog.capture('code_example_copied', {
        code: props.code,
      });
    }
  };

  return (
    <span onClick={handleCopy}>
      <CopyButton {...props} />
    </span>
  );
}
