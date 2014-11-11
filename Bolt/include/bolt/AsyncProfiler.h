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
#if defined(_WIN32)
#pragma once
/******************************************************************************
 * Asynchronous Profiler
 *****************************************************************************/
#include <vector>
#include <Windows.h>


class AsyncProfiler
{
private:
    LARGE_INTEGER constructionTimeStamp;
    LONGLONG timerPeriodNs;
    size_t numThrowAwayTrials;

public:

    static enum attributeTypes {
        /*native*/  id, device, time, memory, bandwidth, flops, flops_s, startTime, stopTime,
        /*total*/   NUM_ATTRIBUTES};
    static char *attributeNames[];// = {"ID", "StartTime", "StopTime", "Memory", "Device", "Flops"};
    static char *trialAttributeNames[];
    /******************************************************************************
     * Class Step
     *****************************************************************************/
    class Step
    {
    private:
        //size_t serial;
        ::std::string stepName;
        size_t attributeValues[NUM_ATTRIBUTES];

    public:

        size_t stdDev[NUM_ATTRIBUTES];
        /******************************************************************************
         * Constructors
         *****************************************************************************/
        Step( );
        ~Step(void);

        /******************************************************************************
         * Member Functions
         *****************************************************************************/
        //void setSerial( size_t s );
        void set( size_t index, size_t value);
        size_t get( size_t index ) const;
        void setName( const ::std::string& name );
        std::string getName( ) const;
        void computeDerived();
        ::std::ostream& writeLog( ::std::ostream& s ) const;

    }; // class Step


    /******************************************************************************
     * Class Trial
     *****************************************************************************/
    class Trial
    {
    private:
        std::vector<Step> steps;
        size_t currentStepIndex;

    public:
        Step aggregateStep;
        std::string trialName;
        size_t attributeValues[NUM_ATTRIBUTES];
        size_t stdDev[NUM_ATTRIBUTES];
        /******************************************************************************
         * Constructors
         *****************************************************************************/
        Trial(void);
        Trial( size_t n );
        ~Trial(void);

        /******************************************************************************
         * Member Functions
         *****************************************************************************/
        void setName( std::string& name );
        size_t size() const;
        void resize( size_t n );
        size_t get( size_t attributeIndex) const;
        size_t get( size_t stepIndex, size_t attributeIndex) const;
        void set( size_t attributeIndex, size_t attributeValue);
        void set( size_t stepIndex, size_t attributeIndex, size_t attributeValue);
        void startStep();
        size_t nextStep();
        void computeStepsDerived();
        void computeAttributes();
        size_t getStepNum() const;
        ::std::ostream& writeLog( ::std::ostream& s ) const;
        void setStepName( const ::std::string& name);
        void setStepName( size_t stepNum, const ::std::string& name);
        std::string getStepName( ) const;
        std::string getStepName( size_t stepNum ) const;
        Step& operator[](size_t idx);
    }; // class Trial


/******************************************************************************
 * Resume Class AsyncProfiler
 *****************************************************************************/
private:
    size_t currentTrialIndex;
    std::vector<Trial> trials;
    std::string name;
    Trial average;
    size_t dataSize;
    std::string architecture;
    size_t trialsAveraged;

public:
    /******************************************************************************
     * Constructors
     *****************************************************************************/
    AsyncProfiler(void);
    AsyncProfiler(std::string name);
    ~AsyncProfiler(void);

    /******************************************************************************
     * Member Functions
     *****************************************************************************/
    size_t getTime();
    void startTrial();
    void stopTrial();
    void nextTrial();
    void nextStep();
    void set( size_t attributeIndex, size_t attributeValue);
    void set( size_t stepIndex, size_t attributeIndex, size_t attributeValue);
    void set( size_t trialIndex, size_t stepIndex, size_t attributeIndex, size_t attributeValue);
    size_t get( size_t attributeIndex) const;
    size_t get( size_t stepIndex, size_t attributeIndex) const;
    size_t get( size_t trialIndex, size_t stepIndex, size_t attributeIndex) const;
    void setStepName( const ::std::string& name);
    size_t getNumTrials() const;
    size_t getNumSteps() const;
    size_t getTrialNum() const;
    size_t getStepNum() const;
    void setName( std::string n);
    void setDataSize( size_t d );
    void setArchitecture( std::string a );
    void throwAway( size_t n);

    void end();
    void calculateAverage();
    ::std::ostream& writeLog( ::std::ostream& s ) const;
    ::std::ostream& writeSum( ::std::ostream& s ) const;
    ::std::ostream& write( ::std::ostream& s ) const;

}; // class AsyncProfiler

#endif
