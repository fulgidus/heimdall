# Dependency Audit Report

**Generated**: 2025-10-25T20:19:05.229696

## Summary

- **Files Processed**: 0
- **Total Packages Locked**: 0
- **Vulnerabilities Found**: 0
- **Errors**: 5

## Errors

- `services/requirements/ml.txt`:   ERROR: Could not find a version that satisfies the requirement torch==2.1.0 (from versions: 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0, 2.7.0, 2.7.1, 2.8.0, 2.9.0)
Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 397, in resolve
    self._add_to_criteria(self.state.criteria, r, parent=None)
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 174, in _add_to_criteria
    raise RequirementsConflicted(criterion)
pip._vendor.resolvelib.resolvers.RequirementsConflicted: Requirements conflict: SpecifierRequirement('torch==2.1.0')

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/resolver.py", line 95, in resolve
    result = self._result = resolver.resolve(
                            ^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 546, in resolve
    state = resolution.resolve(requirements, max_rounds=max_rounds)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 399, in resolve
    raise ResolutionImpossible(e.criterion.information)
pip._vendor.resolvelib.resolvers.ResolutionImpossible: [RequirementInformation(requirement=SpecifierRequirement('torch==2.1.0'), parent=None)]

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/runner/.local/bin/pip-compile", line 8, in <module>
    sys.exit(cli())
             ^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1157, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1078, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1434, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 783, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/decorators.py", line 33, in new_func
    return f(get_current_context(), *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/scripts/compile.py", line 479, in cli
    results = resolver.resolve(max_rounds=max_rounds)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/resolver.py", line 635, in resolve
    is_resolved = self._do_resolve(
                  ^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/resolver.py", line 670, in _do_resolve
    resolver.resolve(
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/resolver.py", line 104, in resolve
    raise error from e
pip._internal.exceptions.DistributionNotFound: No matching distribution found for torch==2.1.0

- `services/requirements/base.txt`: Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 438, in _error_catcher
    yield
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 561, in read
    data = self._fp_read(amt) if not fp_closed else b""
           ^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 527, in _fp_read
    return self._fp.read(amt) if amt is not None else self._fp.read()
                                                      ^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 473, in read
    return self._read_chunked(amt)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 595, in _read_chunked
    while (chunk_left := self._get_chunk_left()) is not None:
                         ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 579, in _get_chunk_left
    chunk_left = self._read_next_chunk_size()
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 539, in _read_next_chunk_size
    line = self.fp.readline(_MAXLINE + 1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/socket.py", line 707, in readinto
    return self._sock.recv_into(b)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/ssl.py", line 1252, in recv_into
    return self.read(nbytes, buffer)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/ssl.py", line 1104, in read
    return self._sslobj.read(len, buffer)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TimeoutError: The read operation timed out

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/.local/bin/pip-compile", line 8, in <module>
    sys.exit(cli())
             ^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1157, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1078, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1434, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 783, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/decorators.py", line 33, in new_func
    return f(get_current_context(), *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/scripts/compile.py", line 479, in cli
    results = resolver.resolve(max_rounds=max_rounds)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/resolver.py", line 635, in resolve
    is_resolved = self._do_resolve(
                  ^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/resolver.py", line 670, in _do_resolve
    resolver.resolve(
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/resolver.py", line 95, in resolve
    result = self._result = resolver.resolve(
                            ^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 546, in resolve
    state = resolution.resolve(requirements, max_rounds=max_rounds)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 397, in resolve
    self._add_to_criteria(self.state.criteria, r, parent=None)
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 173, in _add_to_criteria
    if not criterion.candidates:
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/structs.py", line 156, in __bool__
    return bool(self._sequence)
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 155, in __bool__
    return any(self)
           ^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 143, in <genexpr>
    return (c for c in iterator if id(c) not in self._incompatible_ids)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 44, in _iter_built
    for version, func in infos:
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/factory.py", line 297, in iter_index_candidate_infos
    result = self._finder.find_best_candidate(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/package_finder.py", line 890, in find_best_candidate
    candidates = self.find_all_candidates(project_name)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/package_finder.py", line 831, in find_all_candidates
    page_candidates = list(page_candidates_it)
                      ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/sources.py", line 194, in page_candidates
    yield from self._candidates_from_page(self._link)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/package_finder.py", line 791, in process_project_url
    index_response = self._link_collector.fetch_response(project_url)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/collector.py", line 461, in fetch_response
    return _get_index_content(location, session=self.session)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/collector.py", line 364, in _get_index_content
    resp = _get_simple_response(url, session=session)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/collector.py", line 135, in _get_simple_response
    resp = session.get(
           ^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/sessions.py", line 602, in get
    return self.request("GET", url, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/network/session.py", line 520, in request
    return super().request(method, url, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/cachecontrol/adapter.py", line 76, in send
    resp = super().send(request, stream, timeout, verify, cert, proxies)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/adapters.py", line 538, in send
    return self.build_response(request, resp)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/cachecontrol/adapter.py", line 117, in build_response
    response.read(decode_content=False)
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 560, in read
    with self._error_catcher():
  File "/usr/lib/python3.12/contextlib.py", line 158, in __exit__
    self.gen.throw(value)
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 443, in _error_catcher
    raise ReadTimeoutError(self._pool, None, "Read timed out.")
pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out.

- `services/requirements/data.txt`: Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 438, in _error_catcher
    yield
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 561, in read
    data = self._fp_read(amt) if not fp_closed else b""
           ^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 527, in _fp_read
    return self._fp.read(amt) if amt is not None else self._fp.read()
                                                      ^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 473, in read
    return self._read_chunked(amt)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 595, in _read_chunked
    while (chunk_left := self._get_chunk_left()) is not None:
                         ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 579, in _get_chunk_left
    chunk_left = self._read_next_chunk_size()
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 539, in _read_next_chunk_size
    line = self.fp.readline(_MAXLINE + 1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/socket.py", line 707, in readinto
    return self._sock.recv_into(b)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/ssl.py", line 1252, in recv_into
    return self.read(nbytes, buffer)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/ssl.py", line 1104, in read
    return self._sslobj.read(len, buffer)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TimeoutError: The read operation timed out

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/.local/bin/pip-compile", line 8, in <module>
    sys.exit(cli())
             ^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1157, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1078, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1434, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 783, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/decorators.py", line 33, in new_func
    return f(get_current_context(), *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/scripts/compile.py", line 479, in cli
    results = resolver.resolve(max_rounds=max_rounds)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/resolver.py", line 635, in resolve
    is_resolved = self._do_resolve(
                  ^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/resolver.py", line 670, in _do_resolve
    resolver.resolve(
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/resolver.py", line 95, in resolve
    result = self._result = resolver.resolve(
                            ^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 546, in resolve
    state = resolution.resolve(requirements, max_rounds=max_rounds)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 397, in resolve
    self._add_to_criteria(self.state.criteria, r, parent=None)
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 173, in _add_to_criteria
    if not criterion.candidates:
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/structs.py", line 156, in __bool__
    return bool(self._sequence)
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 155, in __bool__
    return any(self)
           ^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 143, in <genexpr>
    return (c for c in iterator if id(c) not in self._incompatible_ids)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 44, in _iter_built
    for version, func in infos:
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/factory.py", line 297, in iter_index_candidate_infos
    result = self._finder.find_best_candidate(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/package_finder.py", line 890, in find_best_candidate
    candidates = self.find_all_candidates(project_name)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/package_finder.py", line 831, in find_all_candidates
    page_candidates = list(page_candidates_it)
                      ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/sources.py", line 194, in page_candidates
    yield from self._candidates_from_page(self._link)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/package_finder.py", line 791, in process_project_url
    index_response = self._link_collector.fetch_response(project_url)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/collector.py", line 461, in fetch_response
    return _get_index_content(location, session=self.session)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/collector.py", line 364, in _get_index_content
    resp = _get_simple_response(url, session=session)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/collector.py", line 135, in _get_simple_response
    resp = session.get(
           ^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/sessions.py", line 602, in get
    return self.request("GET", url, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/network/session.py", line 520, in request
    return super().request(method, url, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/cachecontrol/adapter.py", line 76, in send
    resp = super().send(request, stream, timeout, verify, cert, proxies)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/adapters.py", line 538, in send
    return self.build_response(request, resp)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/cachecontrol/adapter.py", line 117, in build_response
    response.read(decode_content=False)
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 560, in read
    with self._error_catcher():
  File "/usr/lib/python3.12/contextlib.py", line 158, in __exit__
    self.gen.throw(value)
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 443, in _error_catcher
    raise ReadTimeoutError(self._pool, None, "Read timed out.")
pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out.

- `services/requirements/api.txt`: Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 438, in _error_catcher
    yield
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 561, in read
    data = self._fp_read(amt) if not fp_closed else b""
           ^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 527, in _fp_read
    return self._fp.read(amt) if amt is not None else self._fp.read()
                                                      ^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 473, in read
    return self._read_chunked(amt)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 595, in _read_chunked
    while (chunk_left := self._get_chunk_left()) is not None:
                         ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 579, in _get_chunk_left
    chunk_left = self._read_next_chunk_size()
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 539, in _read_next_chunk_size
    line = self.fp.readline(_MAXLINE + 1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/socket.py", line 707, in readinto
    return self._sock.recv_into(b)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/ssl.py", line 1252, in recv_into
    return self.read(nbytes, buffer)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/ssl.py", line 1104, in read
    return self._sslobj.read(len, buffer)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TimeoutError: The read operation timed out

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/.local/bin/pip-compile", line 8, in <module>
    sys.exit(cli())
             ^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1157, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1078, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1434, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 783, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/decorators.py", line 33, in new_func
    return f(get_current_context(), *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/scripts/compile.py", line 479, in cli
    results = resolver.resolve(max_rounds=max_rounds)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/resolver.py", line 635, in resolve
    is_resolved = self._do_resolve(
                  ^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/resolver.py", line 670, in _do_resolve
    resolver.resolve(
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/resolver.py", line 95, in resolve
    result = self._result = resolver.resolve(
                            ^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 546, in resolve
    state = resolution.resolve(requirements, max_rounds=max_rounds)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 397, in resolve
    self._add_to_criteria(self.state.criteria, r, parent=None)
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 173, in _add_to_criteria
    if not criterion.candidates:
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/structs.py", line 156, in __bool__
    return bool(self._sequence)
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 155, in __bool__
    return any(self)
           ^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 143, in <genexpr>
    return (c for c in iterator if id(c) not in self._incompatible_ids)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 44, in _iter_built
    for version, func in infos:
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/factory.py", line 297, in iter_index_candidate_infos
    result = self._finder.find_best_candidate(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/package_finder.py", line 890, in find_best_candidate
    candidates = self.find_all_candidates(project_name)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/package_finder.py", line 831, in find_all_candidates
    page_candidates = list(page_candidates_it)
                      ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/sources.py", line 194, in page_candidates
    yield from self._candidates_from_page(self._link)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/package_finder.py", line 791, in process_project_url
    index_response = self._link_collector.fetch_response(project_url)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/collector.py", line 461, in fetch_response
    return _get_index_content(location, session=self.session)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/collector.py", line 364, in _get_index_content
    resp = _get_simple_response(url, session=session)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/collector.py", line 135, in _get_simple_response
    resp = session.get(
           ^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/sessions.py", line 602, in get
    return self.request("GET", url, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/network/session.py", line 520, in request
    return super().request(method, url, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/cachecontrol/adapter.py", line 76, in send
    resp = super().send(request, stream, timeout, verify, cert, proxies)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/adapters.py", line 538, in send
    return self.build_response(request, resp)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/cachecontrol/adapter.py", line 117, in build_response
    response.read(decode_content=False)
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 560, in read
    with self._error_catcher():
  File "/usr/lib/python3.12/contextlib.py", line 158, in __exit__
    self.gen.throw(value)
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 443, in _error_catcher
    raise ReadTimeoutError(self._pool, None, "Read timed out.")
pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out.

- `services/requirements/dev.txt`: Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 438, in _error_catcher
    yield
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 561, in read
    data = self._fp_read(amt) if not fp_closed else b""
           ^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 527, in _fp_read
    return self._fp.read(amt) if amt is not None else self._fp.read()
                                                      ^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 473, in read
    return self._read_chunked(amt)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 595, in _read_chunked
    while (chunk_left := self._get_chunk_left()) is not None:
                         ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 579, in _get_chunk_left
    chunk_left = self._read_next_chunk_size()
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/http/client.py", line 539, in _read_next_chunk_size
    line = self.fp.readline(_MAXLINE + 1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/socket.py", line 707, in readinto
    return self._sock.recv_into(b)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/ssl.py", line 1252, in recv_into
    return self.read(nbytes, buffer)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/ssl.py", line 1104, in read
    return self._sslobj.read(len, buffer)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TimeoutError: The read operation timed out

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/.local/bin/pip-compile", line 8, in <module>
    sys.exit(cli())
             ^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1157, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1078, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 1434, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/core.py", line 783, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/click/decorators.py", line 33, in new_func
    return f(get_current_context(), *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/scripts/compile.py", line 479, in cli
    results = resolver.resolve(max_rounds=max_rounds)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/resolver.py", line 635, in resolve
    is_resolved = self._do_resolve(
                  ^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/piptools/resolver.py", line 670, in _do_resolve
    resolver.resolve(
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/resolver.py", line 95, in resolve
    result = self._result = resolver.resolve(
                            ^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 546, in resolve
    state = resolution.resolve(requirements, max_rounds=max_rounds)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 397, in resolve
    self._add_to_criteria(self.state.criteria, r, parent=None)
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/resolvers.py", line 173, in _add_to_criteria
    if not criterion.candidates:
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/resolvelib/structs.py", line 156, in __bool__
    return bool(self._sequence)
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 155, in __bool__
    return any(self)
           ^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 143, in <genexpr>
    return (c for c in iterator if id(c) not in self._incompatible_ids)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 44, in _iter_built
    for version, func in infos:
  File "/usr/lib/python3/dist-packages/pip/_internal/resolution/resolvelib/factory.py", line 297, in iter_index_candidate_infos
    result = self._finder.find_best_candidate(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/package_finder.py", line 890, in find_best_candidate
    candidates = self.find_all_candidates(project_name)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/package_finder.py", line 831, in find_all_candidates
    page_candidates = list(page_candidates_it)
                      ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/sources.py", line 194, in page_candidates
    yield from self._candidates_from_page(self._link)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/package_finder.py", line 791, in process_project_url
    index_response = self._link_collector.fetch_response(project_url)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/collector.py", line 461, in fetch_response
    return _get_index_content(location, session=self.session)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/collector.py", line 364, in _get_index_content
    resp = _get_simple_response(url, session=session)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/index/collector.py", line 135, in _get_simple_response
    resp = session.get(
           ^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/sessions.py", line 602, in get
    return self.request("GET", url, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_internal/network/session.py", line 520, in request
    return super().request(method, url, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/cachecontrol/adapter.py", line 76, in send
    resp = super().send(request, stream, timeout, verify, cert, proxies)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/requests/adapters.py", line 538, in send
    return self.build_response(request, resp)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/pip/_vendor/cachecontrol/adapter.py", line 117, in build_response
    response.read(decode_content=False)
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 560, in read
    with self._error_catcher():
  File "/usr/lib/python3.12/contextlib.py", line 158, in __exit__
    self.gen.throw(value)
  File "/usr/lib/python3/dist-packages/pip/_vendor/urllib3/response.py", line 443, in _error_catcher
    raise ReadTimeoutError(self._pool, None, "Read timed out.")
pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out.


