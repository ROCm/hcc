/***************************************************************************                                                                                     
*   Copyright 2012 - 2013 Advanced Micro Devices, Inc.                                     
*                                                                                    
*   Licensed under the Apache License, Version 2.0 (the "License");   
*   you may not use this file except in compliance with the License.                 
*   You may obtain a copy of the License at                                          
*                                                                                    
*       http://www.apache.org/licenses/LICENSE-2.0                      
*                                                                                    
*   Unless required by applicable law or agreed to in writing, software              
*   distributed under the License is distributed on an "AS IS" BASIS,              
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         
*   See the License for the specific language governing permissions and              
*   limitations under the License.                                                   

***************************************************************************/                                                                                     

// parallel_iteration.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>

#include <amp.h>

#include <bolt/parallel_iteration.h>


//======
struct Match {
	Match(int start=0, int stop=0) : startIndex_(start), stopIndex_(stop) {};

	std::string matchStr(std::string &str) { return str.substr(startIndex_, stopIndex_- startIndex_); };
	int startIndex_;
	int stopIndex_;
};


//======
struct StateNode {
	typedef int IndexType;  // could use a smaller type for indexes, if C++AMP supported it.

	static const int stateSpace = 256;


	StateNode();
	void setDefaultNextState(IndexType nextState);

	// Next state from this node.
	// 0 is the start state
	// -1 indicates we have reached the Final (match) state.
	static const IndexType FinalState = 0;
	static const IndexType StartState = 1;
	IndexType	nextNode_[stateSpace] ;  
};



StateNode::StateNode()
{
	setDefaultNextState(StartState);
};

void StateNode::setDefaultNextState(IndexType nextState)
{
	for (int i=0; i<stateSpace; i++) {
		nextNode_[i] = nextState;
	};
};


//======
class Pattern {
public:
	Pattern(int sz) : states_(sz) {};
	std::vector<StateNode> states_;

	std::vector<Match> matchPattern(std::string str);
	std::vector<Match> matchPatternAll(std::string str);
	std::vector<Match> matchPatternAMP(std::string str);
	std::vector<Match> matchPatternBolt(std::string str);
};


std::vector<Match> Pattern::matchPattern(std::string str)
{
	using namespace std;
	StateNode::IndexType state = StateNode::StartState;
	int startIndex = 0;
	int reachedFinal = 0;
	std::vector<Match> matches;


	for (int j=0; j<(int)str.length(); j++) {
		char c=str[j];
		if (state == StateNode::FinalState) {
			matches.push_back(Match(startIndex, j-1));
			state = StateNode::StartState;
			startIndex = j;
		} else if (state == StateNode::StartState) {
			startIndex = j;
		}
		state = states_[state].nextNode_[c];
		cout << "#"<<j<< "  '" << c << "' ->" << state << (reachedFinal ? "  Final" : "") << endl;
	};
	if (state==StateNode::FinalState) {
		matches.push_back(Match(startIndex, str.length()));
	};
	return matches;
}


//---
// An implementation that examines each character in parallel as a potential start of the string.  
std::vector<Match> Pattern::matchPatternAll(std::string str)
{
	using namespace std;
	StateNode::IndexType state = StateNode::StartState;
	int reachedFinal = 0;
	std::vector<Match> matches;

	for (size_t i=0; i<str.length(); i++) {
		size_t j = i;
		state = StateNode::StartState;
		do {
			char c=str[j];
			state = states_[state].nextNode_[c];
			cout << "#"<<i << "," << j<< ":  '" << c << "' ->" << state << (reachedFinal ? "  Final" : "") << endl;
			j++;
		} while ((state > StateNode::StartState) && (j < str.length()));
		if (state == StateNode::FinalState) {
			cout << "^^^Save" << endl;
			matches.push_back(Match(i, j-1));	
		};
	};

	return matches;
};


std::vector<Match> Pattern::matchPatternAMP(std::string str)
{
	using namespace concurrency;
	const int maxMatches = 16;
	std::vector<Match> matches(maxMatches);

	int charsPerSection = 16; // FIXME.

	// C++AMP does not support chars, as a hacky workaround copy each char to an int.
	// A more optimal implementation would use char4 inside the loop, but lets focus for now on the implementation
	// So we copy each char to a 4-byte int.
	std::vector<int> intStr(str.length());
	for (size_t i=0; i<str.length(); i++) {
		intStr[i] = str[i];
	};


	array_view<StateNode,1> stateV(states_.size(), states_);
	array_view<int,1> strV(intStr.size(), intStr);  
	array_view<Match, 1> matchV(maxMatches, matches);
	array<int,1>  qPtr(1);

	parallel_for_each(strV.grid, [=,&qPtr] (index<1> idx) mutable restrict(direct3d) {
		int state = StateNode::StartState;
		int i = idx[0];

		do {
			int c = strV[i];
			state = stateV[state].nextNode_[c];
			i++;

		} while (state>StateNode::StartState && (i < strV.grid.extent[0]));


		if (state == StateNode::FinalState) {
			int lPtr = atomic_fetch_add(&qPtr[0], 1);
			matchV[lPtr].startIndex_ = idx[0];
			matchV[lPtr].stopIndex_  = i-1;
		};
	});

	return matches;
};



//---
// Bolt implementation - enable developer to focus on just the iteration.
std::vector<Match> Pattern::matchPatternBolt(std::string str)
{	
	using namespace concurrency;
	const int maxMatches = 16;
	std::vector<Match> matches(maxMatches);

	// C++AMP does not support chars, as a hacky workaround copy each char to an int.
	// A more optimal implementation would use char4 inside the loop, but lets focus for now on the implementation
	// So we copy each char to a 4-byte int.
	std::vector<int> intStr(str.length());
	for (size_t i=0; i<str.length(); i++) {
		intStr[i] = str[i];
	};
	const int strLength = intStr.size();

	array_view<StateNode,1> stateV(states_.size(), states_);
	array_view<int,1> strV(intStr.size(), intStr);  
	array_view<Match, 1> matchV(maxMatches, matches);
	array<int,1>  qPtr(1);

	// This structure captures the state needed for each iteration to keep track of where we are.
	struct IterationState {
		IterationState() : ci(0), state(StateNode::StartState)  {};

		int ci;				// current char we are examining.
		StateNode::IndexType  state; // last state.
	};


	IterationState initState;


	bolt::parallel_iteration(extent<1>(intStr.size()), initState, [=,&qPtr](index<1> idx, IterationState &iter) restrict(direct3d) ->bool {

		int c = strV[idx+iter.ci];
		iter.state = stateV[iter.state].nextNode_[c];
		iter.ci++;

		if ((iter.state > StateNode::StartState) && (idx[0]+iter.ci < strLength)) {
			return true; // keep searching
		} else {
#if 1
			if (iter.state == StateNode::FinalState) {
				int lPtr = atomic_fetch_add(&qPtr[0], 1);
				if (lPtr < maxMatches) {
					matchV[lPtr].startIndex_ = idx[0];
					matchV[lPtr].stopIndex_  = iter.ci - 1;
				}
			};
#endif
			return false;  // done searching.
		}


	});



	return matches;
};


//======
int _tmain(int argc, _TCHAR* argv[])
{
	using namespace std;
	Pattern p(1024);


	int state = 0; // final state, stay here if we get here
	p.states_[state].setDefaultNextState(StateNode::FinalState);

	state = 1; // state 1 consumes leading whitespace
	p.states_[state].setDefaultNextState(2);
	p.states_[state].nextNode_[' '] = state;
	p.states_[state].nextNode_['\t'] = state;
	p.states_[state].nextNode_['f'] = 3;  // repeating 'f' 
	p.states_[state].nextNode_['r'] = 4;  // repeating 'r' 


	state=2;  // Look for lower-case chars.
	for (char c='a'; c<'z'; c++) {
		p.states_[state].nextNode_[c] = state;
	};
	p.states_[state].nextNode_['f'] = 3;  // repeating 'f' 
	p.states_[state].nextNode_['r'] = 4;  // repeating 'r' 

	state=3; // check for repeating 'f'
	p.states_[state].setDefaultNextState(1);
	p.states_[state].nextNode_['f'] = 5;
	p.states_[state].nextNode_['r'] = 4;  // catch case of 'f' followed by 'r'

	state=4; // check for repeating 'r'
	p.states_[state].setDefaultNextState(1);
	p.states_[state].nextNode_['r'] = 5;
	p.states_[state].nextNode_['f'] = 3;  // catch case of 'r' followed by 'f'

	state=5;  // Repeat detected, consume more chars to complete word.
	p.states_[state].setDefaultNextState(StateNode::FinalState);
	for (char c='a'; c<'z'; c++) {
		p.states_[state].nextNode_[c] = state;
	};


	std::string str = "The quick fox offers his paw in an array.    Tomorrow we should dance.";

	{
		//vector<Match> matches = p.matchPattern(str);
		vector<Match> matches = p.matchPatternAll(str);
		for_each (matches.begin(), matches.end(), [&] (Match m) {
			std::cout << "match result: (" << m.startIndex_ << "," << m.stopIndex_ << ") <" <<  m.matchStr(str) << '>' << std::endl;
		});
	}

	{
		cout << "\nAMP\n";
		vector<Match> matchesAmp = p.matchPatternAMP(str);
		for_each (matchesAmp.begin(), matchesAmp.end(), [&] (Match m) {
			std::cout << "match result: (" << m.startIndex_ << "," << m.stopIndex_ << ") <" <<  m.matchStr(str) << '>' << std::endl;
		});
	}

	{
		cout << "\nBolt\n";
		vector<Match> matchesBolt = p.matchPatternBolt(str);
		for_each (matchesBolt.begin(), matchesBolt.end(), [&] (Match m) {
			std::cout << "match result: (" << m.startIndex_ << "," << m.stopIndex_ << ") <" <<  m.matchStr(str) << '>' << std::endl;
		});
	}
};



// Demand paging - only copy GPU data as-needed.


