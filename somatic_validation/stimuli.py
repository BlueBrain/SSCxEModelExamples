"""Stimuli classes"""

"""
Copyright (c) 2022 EPFL-BBP, All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE BLUE BRAIN PROJECT
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/legalcode or send a letter to Creative Commons, 171
Second Street, Suite 300, San Francisco, California, 94105, USA.
"""

# pylint: disable=W0511

import logging

logger = logging.getLogger(__name__)


class Stimulus(object):

    """Stimulus protocol"""

    pass


class NrnHDPulse(Stimulus):

    """Square pulse current clamp injection"""

    def __init__(
        self,
        step_amplitude=None,
        step_delay=None,
        step_duration=None,
        total_duration=None,
        depol=None,
        duration_of_depol1=None,
        duration_of_depol2=None,
        # hold=None
        location=None,
    ):
        """Constructor

        Args:
            step_amplitude (float): amplitude (nA)
            step_delay (float): delay (ms)
            step_duration (float): duration (ms)
            total_duration (float): total duration (ms)
            depol (float): amplitude (nA)
            durration_of_depol (float): duration of depolarization (ms)
            location (Location): stimulus Location
        """

        super(NrnHDPulse, self).__init__()
        self.step_amplitude = step_amplitude
        self.step_delay = step_delay
        self.step_duration = step_duration
        self.location = location
        self.total_duration = total_duration
        self.depol = depol
        self.duration_of_depol1 = duration_of_depol1
        self.duration_of_depol2 = duration_of_depol2
        # self.hold=None
        self.iclamp = None
        self.persistent = []  # TODO move this into higher abstract classes

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        icomp = self.location.instantiate(sim=sim, icell=icell)
        logger.debug(
            "Adding square step stimulus to %s with delay %f, "
            "duration %f, and amplitude %f",
            str(self.location),
            self.step_delay,
            self.step_duration,
            self.step_amplitude,
        )

        # create vector to store the times at which stim amp changes
        times = sim.neuron.h.Vector()
        # create vector to store to which stim amps over time
        amps = sim.neuron.h.Vector()

        # at time 0.0, current is 0.0
        times.append(0.0)
        amps.append(0.0)

        # until time ramp_delay, current is 0.0
        times.append(self.step_delay)
        amps.append(0.0)

        # at time ramp_delay, current is ramp_amplitude_start
        times.append(self.step_delay)
        amps.append(self.depol)

        # at time ramp_delay+ramp_duration, current is ramp_amplitude_end
        times.append(self.step_delay + self.duration_of_depol1)
        amps.append(self.depol)

        # after ramp, current is set 0.0
        times.append(self.step_delay + self.duration_of_depol1)
        amps.append(self.step_amplitude - self.depol)

        times.append(self.step_delay + self.step_duration - self.duration_of_depol2)
        amps.append(self.step_amplitude - self.depol)

        times.append(self.step_delay + self.step_duration - self.duration_of_depol2)
        amps.append(self.depol)

        times.append(self.step_delay + self.step_duration)
        amps.append(self.depol)

        times.append(self.step_delay + self.step_duration)
        amps.append(0.0)

        times.append(self.total_duration)
        amps.append(0.0)

        # create a current clamp
        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = self.total_duration

        # play the above current amplitudes into the current clamp
        amps.play(self.iclamp._ref_amp, times, 1)  # pylint: disable=W0212

        # Make sure the following objects survive after instantiation
        self.persistent.append(times)
        self.persistent.append(amps)

    def destroy(self, sim=None):
        """Destroy stimulus"""
        self.persistent = []

        self.iclamp = None

    def __str__(self):
        """String representation"""

        return "Square pulse amp %f delay %f duration %f totdur %f at %s" % (
            self.step_amplitude,
            self.step_delay,
            self.step_duration,
            self.total_duration,
            self.depol,
            self.duration_of_depol1,
            self.duration_of_depol2,
            self.location,
        )
