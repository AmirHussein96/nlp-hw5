#!/usr/bin/env python3
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

# Recognizer code by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# This code is hereby released to the public domain.

from __future__ import annotations
import argparse
import logging
import math
import tqdm
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple
import math
import pdb
import os
from copy import deepcopy

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol", 
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    parser.add_argument(
        "--progress", 
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    args = parser.parse_args()
    return args


class EarleyChart:
    """A chart for Earley's algorithm."""
    
    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.  
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()
        self.min_parse_weight = math.inf # Amir: start with inf weight
        self.best_attached = {} # dictionary of all the attached elements
        self.tobe_attached = {} # dictionary of possible parents for attachment
        self.root = None
        self.cols: List[Agenda]
        self.check_duplicates = {}
        self.tobe_processed = {}
        self.traverse_output = "" # Don string for printing
        self._run_earley()    # run Earley's algorithm to construct self.cols
        
        

    def accepted(self) -> bool:
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        for item in self.cols[-1].all():    # the last column
            if (item.rule.lhs == self.grammar.start_symbol   # a ROOT item in this column
                and item.next_symbol() is None               # that is complete 
                and item.start_position == 0):               # and started back at position 0
                    return self.root
        return False   # we didn't find any appropriate item

    def _run_earley(self) -> None:
        """Fill in the Earley chart"""
        # Initially empty column for each position in sentence
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]

        # Start looking for ROOT at position 0
       # pdb.set_trace()
        self._predict(self.grammar.start_symbol, 0)
        
        # We'll go column by column, and within each column row by row.
        # Processing earlier entries in the column may extend the column
        # with later entries, which will be processed as well.
        # 
        # The iterator over numbered columns is `enumerate(self.cols)`.  
        # Wrapping this iterator in the `tqdm` call provides a progress bar.
        for i, column in tqdm.tqdm(enumerate(self.cols),
                                   total=len(self.cols),
                                   disable=not self.progress):
            logging.debug("")
            logging.debug(f"Processing items in column {i}")
            self.check_processed ={}
            #self.check_duplicates = {}
            while column:    # while agenda isn't empty
                
                item = column.pop()   # dequeue the next unprocessed item
                next = item.next_symbol()
                # print(item.rule)
                # print(next)
                if next is None:
                    # Attach this complete constituent to its customers
                    logging.debug(f"{item} => ATTACH")
                    self._attach(item, i)   
                elif self.grammar.is_nonterminal(next):
                    # Predict the nonterminal after the dot
                    logging.debug(f"{item} => PREDICT")
                    self._predict(next, i)
                else:
                    # Try to scan the terminal after the dot
                    logging.debug(f"{item} => SCAN")
                    self._scan(item, i)                      

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position."""
        for rule in self.grammar.expansions(nonterminal):   # this looks into all possibple rules for the nonterminal (need to check if this has been precessed for efficiency)
            new_item = Item(rule, dot_position=0, start_position=position)   
            
            # if (position, new_item.rule.lhs, new_item.rule.rhs)  not in self.check_duplicates:
            #     self.check_duplicates[(position, new_item.rule.lhs, new_item.rule.rhs)] = True
            self.cols[position].push(new_item)
            
            logging.debug(f"\tPredicted: {new_item} in column {position}")
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:

        """Attach the next word to this item that ends at position, 
        if it matches what this item is looking for next."""
        # pdb.set_trace()
        # print(item.next_symbol())
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            new_item = item.with_dot_advanced()
            self.cols[position + 1].push(new_item)
            logging.debug(f"\tScanned to get: {new_item} in column {position+1}")
            self.profile["SCAN"] += 1
            
            if  (new_item.start_position, new_item.rule.lhs, new_item.rule.rhs, new_item.dot_position-1) in self.tobe_processed:
                # pdb.set_trace()
                updated_nodes = self.tobe_processed[(new_item.start_position, new_item.rule.lhs, new_item.rule.rhs, new_item.dot_position-1)]
                for updated_node in updated_nodes:
                    updated_node.dot_position = new_item.dot_position
                self.tobe_processed[(new_item.start_position, new_item.rule.lhs, new_item.rule.rhs, new_item.dot_position)] = updated_nodes
               # del self.tobe_attached[(new_item.start_position, new_item.rule.lhs, new_item.rule.rhs, new_item.dot_position-1)]
                if new_item.dot_position == len(new_item.rule.rhs):
                    self.best_attached[(updated_node.name, updated_node.rule.rhs,updated_node.start_position,position+1)] = updated_nodes[0] 
                #    del self.tobe_attached[(new_item.start_position, new_item.rule.lhs, new_item.rule.rhs, new_item.dot_position)]
            elif new_item.dot_position == len(new_item.rule.rhs) and len(new_item.rule.rhs) == 1: # add the unary rules to the graph
                    updated_node = Node(new_item,item.rule.lhs,position+1)
                    self.best_attached[(updated_node.name, updated_node.rule.rhs, updated_node.start_position,position+1)] = updated_node 

            
    def _attach(self, item: Item, position: int) -> None:
        """Attach this complete item to its customers in previous columns, advancing the
        customers' dots to create new items in this column.  (This operation is sometimes
        called "complete," but actually it attaches an item that was already complete.)
        """
        mid = item.start_position   # start position of this item = end position of item to its left
        for customer in self.cols[mid].all():  # could you eliminate this inefficient linear search? # this searches for all items in the column
            if customer.next_symbol() == item.rule.lhs:
                new_item = customer.with_dot_advanced()
                self.cols[position].push(new_item)
                logging.debug(f"\tAttached to get: {new_item} in column {position}")
                self.profile["ATTACH"] += 1
                #if customer not in self.check_processed:
                #self.check_processed[customer]=True
                    # for rule in customer.rules:
                    # convert item to a node which we can update
                    # print(self.best_attached)
                    #print('customer: ' ,new_item)
                    #print(position)
                    #pdb.set_trace()
                    
                    #if position == 3 and new_item.rule.lhs =='VP' and new_item.rule.rhs ==('VBZ','ADJP-PRD'):
                # if position == 3 and new_item.rule.lhs =='S' and new_item.rule.rhs ==('NP','VP','PUNC.'):
                # if position == 13:
                #     print('customer: ' ,new_item)
                #     print('item: ' ,item)
                #     print(position)
                #     pdb.set_trace()
                if (item.rule.lhs, item.rule.rhs, item.start_position, position) not in self.best_attached:
                    node_item = Node(item,item.rule.lhs, position)
                    node_customers = self.get_parent(new_item, position,node_item)
                else:
                    if item.rule.weight < self.best_attached[(item.rule.lhs, item.rule.rhs, item.start_position,position)].weight: # check the minimum weight
                        node_item = Node(item,item.rule.lhs, position)
                        node_item.update_connections(self.best_attached[(item.rule.lhs, item.rule.rhs,item.start_position,position)])
                        
                        node_customers = self.get_parent(new_item, position,node_item)

                    else:
                        
                        node_item = self.best_attached[(item.rule.lhs, item.rule.rhs,item.start_position,position)] 
                        node_customers = self.get_parent(new_item, position,node_item)

                for node_customer in node_customers:
                    if node_customer != None:
                        if (node_item not in node_customer.children):
                        
                            self.add_to_graph(node_item, node_customer, mid, position)
                    #     self.best_attached[(customer,mid,position)]={new_item.rule.weight:new_item} # key of triplet (X,I,J) and the coresponding cost to create it 
        
    def get_parent(self, customer, end_position, child):
        # if (customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position) == (0, 'ADVP', ('ADVP',), 1):
        #     pdb.set_trace()
        if (customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position) in self.check_processed and end_position > 1:
           
            # if self.check_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)].children:
            #     if child.total_weight <  self.check_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)].children[-1].total_weight:
            #                 node_customer = self.check_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)]
            #                 node_customer.total_weight -= node_customer.children[-1].total_weight
            #                 node_customer.children.pop()
            #                 self.check_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)] = node_customer
            #                 self.tobe_attached[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)] = node_customer

            #     else:
            #         return None
            # else:
            #     return None
            #print((customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position))
            
            # get the previous node since it has been processed
            node_customers = deepcopy(self.tobe_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)])
            for x in node_customers:
                x.total_weight -= x.children[-1].total_weight
                x.children.pop()
                self.tobe_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)].append(x)
            
        elif (customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position-1) in self.tobe_processed:
            # if the parent is in the temporary dict update its position
            if customer.dot_position == len(customer.rule.rhs):
                node_customers = deepcopy(self.tobe_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position-1)])
                
                # remove the old one 
                #del self.tobe_attached[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position-1)]
                for x in node_customers:
                    x.dot_position = customer.dot_position
                    #self.tobe_attached[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)] = node_customer
                    self.check_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)] = x
                self.tobe_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)] = node_customers
            else:
                node_customers = deepcopy(self.tobe_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position-1)])
                
                # remove the old one 
                #del self.tobe_attached[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position-1)]
                for x in node_customers:
                    x.dot_position = customer.dot_position
                    self.check_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)] = x
                self.tobe_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)] = node_customers

        else:
            node_customers = [Node(customer,customer.rule.lhs, customer.dot_position)]
            self.tobe_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)] = node_customers
            self.check_processed[(customer.start_position,customer.rule.lhs, customer.rule.rhs, customer.dot_position)] = node_customers
        return node_customers
    

    # def clean_root(self, root):
    #     if (root.name,root.start_position,root.end_position) in self.best_attached:
    #         del self.best_attached[(root.name,root.start_position,root.end_position)]
    #         for child in root.children:
    #             self.clean_root(child)
    #     else:
    #         return
        
       

    def add_to_graph(self, child, parent, startpos, endpos):
      #  if parent.name == 'ROOT':
      #      pdb.set_trace()
        # if not self.grammar.is_nonterminal(child.name):
        #     if (child.name, startpos, endpos) not in self.best_attached:
        #         self.best_attached[(child.name,startpos,endpos)] = child  # key of triplet (X,I,J) and the coresponding cost to create it
        #         child.add_parent(parent)
        #     elif (child.name, startpos, endpos) in self.best_attached:
        #         if child.weight < self.best_attached[(child.name,startpos,endpos)].weight: # check the minimum weight
        #             self.best_attached[(child.name,startpos,endpos)] = child
        #             child = self.best_attached[(child.name,startpos,endpos)] 

        #     if parent.dot_position == len(parent.rule.rhs): # if the dot reached the end 
        #         if (parent.name, startpos, endpos) not in self.best_attached:
        #             self.best_attached[(parent.name,startpos,endpos)] = parent 
        #         parent.add_children(child)
        #         #parent.total_weight+= parent.weight
        #         parent.end_position = endpos
        # else:
        if (child.name, child.rule.rhs, startpos, endpos) not in self.best_attached:
            self.best_attached[(child.name,startpos,endpos)] = child
            child.add_parent(parent) 
            parent.total_weight+= child.total_weight
            parent.add_children(child)

        elif (child.name, child.rule.rhs, startpos, endpos) in self.best_attached:
            if child.weight < self.best_attached[(child.name,child.rule.rhs, startpos,endpos)].weight: # check the minimum weight
                self.best_attached[(child.name, child.rule.rhs,startpos,endpos)] = child
            #   else:
            #       child = self.best_attached[(child.name,startpos,endpos)]
            #child.parent = parent 
            parent.total_weight+= child.total_weight
            parent.add_children(child)  
                              
        if parent.dot_position == len(parent.rule.rhs): # if the dot reached the end 
            # if parent.name == self.grammar.start_symbol and parent.end_position < len(self.tokens):
            #     self.clean_root(parent)

            if (parent.name,parent.rule.rhs, parent.start_position , endpos) not in self.best_attached:
               # parent.total_weight+= parent.weight  # we start the total weight with the node rule weight, so no need to add again
                parent.end_position = endpos
              #  pdb.set_trace()
                self.best_attached[(parent.name, parent.rule.rhs,parent.start_position,endpos)] = parent 
            elif (parent.name, parent.rule.rhs, parent.start_position , endpos) in self.best_attached:
                if parent.weight < self.best_attached[(parent.name, parent.rule.rhs,parent.start_position,endpos)].weight: # check the minimum weight
                   # parent.total_weight+= parent.weight
                    self.best_attached[(parent.name, parent.rule.rhs, parent.start_position,endpos)] = parent 

            # if (parent.start_position, parent.rule.lhs, parent.rule.rhs, parent.dot_position) in self.tobe_processed:
            #     del self.tobe_processed[(parent.start_position, parent.rule.lhs, parent.rule.rhs, parent.dot_position)] # remove from temporary dict
                
        if parent.name == self.grammar.start_symbol and len(self.tokens) == parent.end_position:
            if parent.total_weight < self.min_parse_weight:
                self.min_parse_weight = parent.total_weight
               # pdb.set_trace()
                self.root = parent  

    def traverse(self, node, position):
        if len(self.traverse_output) > 0 and self.traverse_output[-1] == ")":
            self.traverse_output += " "
        self.traverse_output = self.traverse_output + "(" + node.name + " "
        # Don dictionary of children for printing tree
        for child in node.children:
            node.child_dict[child.start_position] = (child, child.end_position)

        node.print_loc
        for thing in node.rule.rhs:
            # pdb.set_trace()
            if not self.grammar.is_nonterminal(thing):
                self.traverse_output += thing
                node.print_loc += 1
            else:
           #     pdb.set_trace()
                self.traverse(node.child_dict[node.print_loc][0], node.print_loc)
                node.print_loc = node.child_dict[node.print_loc][1]
        
        self.traverse_output += ")"
        return self.traverse_output
    
    

class Agenda:
    """An agenda of items that need to be processed.  Newly built items 
    may be enqueued for processing by `push()`, and should eventually be 
    dequeued by `pop()`.

    This implementation of an agenda also remembers which items have
    been pushed before, even if they have subsequently been popped.
    This is because already popped items must still be found by
    duplicate detection and as customers for attach.  

    (In general, AI algorithms often maintain a "closed list" (or
    "chart") of items that have already been popped, in addition to
    the "open list" (or "agenda") of items that are still waiting to pop.)

    In Earley's algorithm, each end position has its own agenda -- a column
    in the parse chart.  (This contrasts with agenda-based parsing, which uses
    a single agenda for all items.)

    Standardly, each column's agenda is implemented as a FIFO queue
    with duplicate detection, and that is what is implemented here.
    However, other implementations are possible -- and could be useful
    when dealing with weights, backpointers, and optimizations.

    >>> a = Agenda()
    >>> a.push(3)
    >>> a.push(5)
    >>> a.push(3)   # duplicate ignored
    >>> a
    Agenda([]; [3, 5])
    >>> a.pop()
    3
    >>> a
    Agenda([3]; [5])
    >>> a.push(3)   # duplicate ignored
    >>> a.push(7)
    >>> a
    Agenda([3]; [5, 7])
    >>> while a:    # that is, while len(a) != 0
    ...    print(a.pop())
    5
    7

    """

    def __init__(self) -> None:
        self._items: List[Item] = []       # list of all items that were *ever* pushed
        self._next = 0                     # index of first item that has not yet been popped
        self._index: Dict[Item, int] = {}  # stores index of an item if it has been pushed before

        # Note: There are other possible designs.  For example, self._index doesn't really
        # have to store the index; it could be changed from a dictionary to a set.  
        # 
        # However, we provided this design because there are multiple reasonable ways to extend
        # this design to store weights and backpointers.  That additional information could be
        # stored either in self._items or in self._index.

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables `len(my_agenda)`."""
        return len(self._items) - self._next

    def push(self, item: Item) -> None:
        """Add (enqueue) the item, unless it was previously added."""
        if item not in self._index:    # O(1) lookup in hash table
            self._items.append(item)
            self._index[item] = len(self._items) - 1
            
    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        if len(self)==0:
            raise IndexError
        item = self._items[self._next]
        self._next += 1
        return item

    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed, even if 
        they've already been popped."""
        return self._items

    def __repr__(self):
        """Provide a REPResentation of the instance for printing."""
        next = self._next
        return f"{self.__class__.__name__}({self._items[:next]}; {self._items[next:]})"

class Grammar:
    """Represents a weighted context-free grammar."""
    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol, 
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}    # maps each LHS to the list of rules that expand it
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited linfore of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())  
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions


# A dataclass is a class that provides some useful defaults for you. If you define
# the data that the class should hold, it will automatically make things like an
# initializer and an equality function.  This is just a shortcut.  
# More info here: https://docs.python.org/3/library/dataclasses.html
# Using a dataclass here lets us specify that instances are "frozen" (immutable),
# and therefore can be hashed and used as keys in a dictionary.
@dataclass(frozen=True)
class Rule:
    """
    Convenient abstraction for a grammar rule. 
    A rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        return f"{self.lhs} → {' '.join(self.rhs)}"

class Node:
    """Node for parse graph"""
    def __init__(self,item,name,end_position):
        self.rule = item.rule
        self.dot_position = item.dot_position
        self.start_position = item.start_position
        self.end_position = end_position
        #self.customer = None # the head/parent of the node
        self.name = name   # Amir: the head of the node
        self.total_weight = item.rule.weight
        self.children = []
        self.parents = {}
        self.weight = item.rule.weight
        if self.name == 'ROOT':
            self.isroot = True
        else:
            self.isroot = False
        self.child_dict = {} # Don dictionary of children for printing tree
        self.print_loc = self.start_position # Don keep track of printing
                
    def add_children(self,node):
        self.children.append(node)

    def add_parent(self,node):
        self.parents[node.name]=node

    def update_connections(self, item):
        self.total_weight = item.total_weight - item.rule.weight + self.weight
        if item.children: # Amir: inherit all the children from item
            self.children = item.children
            
        if item.parents:
            self.parent = item.parent

    def print_children(self):
        for child in self.children:
            print('child: ', child.name, ' with rule: ', child.start_position, '-' , child.end_position , child.rule)
# We particularly want items to be immutable, since they will be hashed and 
# used as keys in a dictionary (for duplicate detection).  
@dataclass(frozen=True)
class Item:
    """An item in the Earley parse table, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    dot_position: int
    start_position: int
    # We don't store the end_position, which corresponds to the column
    # that the item is in, although you could store it redundantly for 
    # debugging purposes if you wanted.

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self) -> Item:
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
             
        return Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position)

    def __repr__(self) -> str:
        """Complete string used to show this item at the command line"""
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {dotted_rule})"  # matches notation on slides


def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.verbose)  # Set logging level appropriately
   # pdb.set_trace()
    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                # analyze the sentence
                logging.debug("="*70)
                logging.debug(f"Parsing sentence: {sentence}")
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                # print the result
                # print(
                #     f"'{sentence}' is {'accepted' if chart.accepted() else 'rejected'} by {args.grammar}"
                # )
                logging.debug(f"Profile of work done: {chart.profile}")
                if chart.accepted():
                    tree_parse = chart.traverse(chart.accepted(), 0)
                    t = os.system(f"echo '{tree_parse}' | ./prettyprint")
                    print(t)
                    print(chart.root.total_weight)
                else:
                    print('NONE')


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)   # run tests
    main()
