/*  Axtor - AST-Extractor for LLVM
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
/*
 * Timer.h
 *
 *  Created on: Feb 24, 2011
 *      Author: Simon Moll
 */

#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <sys/time.h>

namespace axtor {

	class Timer
	{
		timeval startTime;
		timeval endTime;

		void dumpTV(timeval v) const
		{
			std::cerr << "secs " << v.tv_sec << "\n"
					  << "ms   " << v.tv_usec << "\n";
		}

	public:
		inline void start()
		{
			timespec start;
			clock_gettime(CLOCK_REALTIME, &start);
			TIMESPEC_TO_TIMEVAL(&startTime, &start)
		}

		inline void end()
		{
			timespec end;
			clock_gettime(CLOCK_REALTIME, &end);
			TIMESPEC_TO_TIMEVAL(&endTime, &end)
		}

		long getTotalMS() const
		{
			long start = startTime.tv_sec * 1000 + (startTime.tv_usec / 1000);
			long end = endTime.tv_sec * 1000 + (endTime.tv_usec / 1000);

			return end - start;
		}
	};

}

#endif /* TIMER_HPP_ */
