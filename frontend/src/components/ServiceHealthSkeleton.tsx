import React from 'react';
import Skeleton from './Skeleton';

/**
 * Skeleton loader for Services Status section
 */
const ServiceHealthSkeleton: React.FC = () => {
    return (
        <ul className="list-group list-group-flush">
            {[1, 2, 3, 4, 5].map((index) => (
                <li key={index} className="list-group-item px-0">
                    <div className="d-flex align-items-center justify-content-between">
                        <div className="flex-grow-1">
                            <Skeleton width="120px" height="20px" />
                        </div>
                        <div className="flex-shrink-0">
                            <Skeleton width="70px" height="24px" />
                        </div>
                    </div>
                </li>
            ))}
        </ul>
    );
};

/**
 * Skeleton loader for WebSDR cards
 */
export const WebSDRCardSkeleton: React.FC = () => {
    return (
        <>
            {[1, 2, 3, 4, 5, 6, 7].map((index) => (
                <div key={index} className="col-lg-3 col-md-4 col-sm-6">
                    <div className="card bg-light border-0 mb-3">
                        <div className="card-body">
                            <div className="d-flex align-items-center justify-content-between mb-2">
                                <Skeleton width="80px" height="20px" />
                                <Skeleton width="32px" height="32px" variant="circular" />
                            </div>
                            <Skeleton width="60px" height="16px" className="mb-2" />
                            <div className="d-flex align-items-center">
                                <div className="flex-grow-1 me-2">
                                    <Skeleton height="5px" />
                                </div>
                                <Skeleton width="30px" height="16px" />
                            </div>
                        </div>
                    </div>
                </div>
            ))}
        </>
    );
};

/**
 * Skeleton loader for stat cards
 */
export const StatCardSkeleton: React.FC = () => {
    return (
        <div className="card">
            <div className="card-body">
                <Skeleton width="100px" height="20px" className="mb-4" />
                <div className="row d-flex align-items-center">
                    <div className="col-9">
                        <Skeleton width="60px" height="32px" />
                    </div>
                    <div className="col-3 text-end">
                        <Skeleton width="40px" height="20px" />
                    </div>
                </div>
                <div className="m-t-30">
                    <Skeleton height="7px" />
                </div>
            </div>
        </div>
    );
};

export default ServiceHealthSkeleton;
