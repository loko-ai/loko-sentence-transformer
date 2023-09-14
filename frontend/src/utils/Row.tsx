import { Flex, FlexProps } from '@chakra-ui/layout';
import React from 'react';
import { ReactNode } from 'react';

export interface RowProps extends FlexProps {
  children: ReactNode;
}

export function Row({ children, ...rest }: RowProps) {
  return (
    <Flex
      p={3}
      mb={0.25}
      w="full"
      cursor="pointer"
      alignItems="center"
      justifyContent="space-between"
      _focus={{ boxShadow: '0 0 8px #76E4F7' }}
      style={{ transition: 'background 200ms ease-in-out' }}
      {...rest}
    >
      {children}
    </Flex>
  );
}

export function RowBody({ children, ...rest }: RowProps) {
  return (
    <Flex flexDir="column" {...rest}>
      {children}
    </Flex>
  );
}

export function RowActions({ children, ...rest }: RowProps) {
  return (
    <Flex flexDir="row" alignItems="center" {...rest}>
      {children}
    </Flex>
  );
}
